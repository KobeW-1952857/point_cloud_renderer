#include "glm/fwd.hpp"
#include "glm/matrix.hpp"
#include "happly.h"
#include "render.cuh"

#include <cstdint>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/types.h>

struct Camera {
  glm::mat3 intrinsic = glm::mat3(1.0f);
  std::vector<float> distortion;
  uint width, height;
  uint id;
};

bool readCameraFile(const std::string &file_path, Camera &cam) {
  std::ifstream configurationFile(file_path);
  if (!configurationFile.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(configurationFile, line)) {
    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream iss(line);

    std::string model;
    float fx, fy, cx, cy;
    float k1, k2, p1, p2;

    iss >> cam.id >> model >> cam.width >> cam.height >> fx >> fy >> cx >> cy >>
        k1 >> k2 >> p1 >> p2;

    cam.intrinsic[0][0] = fx;
    cam.intrinsic[1][1] = fy;
    cam.intrinsic[0][2] = cx;
    cam.intrinsic[1][2] = cy;

    cam.distortion.push_back(k1);
    cam.distortion.push_back(k2);
    cam.distortion.push_back(p1);
    cam.distortion.push_back(p2);

    break;
  }

  return true;
} // Function to print a glm::mat4
void printMat4(const glm::mat4 &mat) {
  // Set output formatting for floating-point numbers
  std::cout << std::fixed << std::setprecision(4);

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      // GLM stores matrices in column-major order (like OpenGL)
      // So mat[column][row] is the correct way to access elements.
      // When printing, we usually want to display it row by row.
      std::cout << std::setw(10) << mat[j][i] << " ";
    }
    std::cout << std::endl;
  }
}

bool readExtrinsics(const std::string &file_path,
                    std::vector<glm::mat4> &extrinsics) {

  std::ifstream configurationFile(file_path);
  if (!configurationFile.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(configurationFile, line)) {
    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream iss(line);

    glm::quat rotation;
    glm::vec3 translation;
    uint id;

    iss >> id >> rotation.w >> rotation.x >> rotation.y >> rotation.z >>
        translation.x >> translation.y >> translation.z;

    glm::mat4 extrinsic = glm::mat4_cast(rotation);
    extrinsic[3][0] = translation.x;
    extrinsic[3][1] = translation.y;
    extrinsic[3][2] = translation.z;

    extrinsics.push_back(extrinsic);
  }

  return true;
}
// Function to save raw RGBA pixel data to a PPM image file
void savePPMImage(const std::string &filename, const unsigned char *data,
                  int width, int height, bool binary = true) {
  std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }

  // Write PPM header
  if (binary) {
    file << "P3\n";
  } else {
    file << "P6\n";
  }

  file << width << " " << height << "\n";
  file << "255\n"; // Max color value

  // Write pixel data (only RGB, skip Alpha for PPM P6)
  for (int i = 0; i < width * height; ++i) {
    if (binary) {
      file.write(reinterpret_cast<const char *>(&data[i * 3 + 0]), 1); // Red
      file.write(reinterpret_cast<const char *>(&data[i * 3 + 1]), 1); // Green
      file.write(reinterpret_cast<const char *>(&data[i * 3 + 2]), 1); // Blue
    } else {
      file << std::to_string(data[i * 3 + 0]) << " "
           << std::to_string(data[i * 3 + 1]) << " "
           << std::to_string(data[i * 3 + 2]) << "\n";
    }
  }

  file.close();
  std::cout << "Image saved to " << filename << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cerr << "Wrong arguments suplied\nUsage: " << argv[0]
              << " <data_path> <camera_path> <extrinsics_path> <output_folder>"
              << std::endl;
    return 1;
  }

  happly::PLYData point_cloud(argv[1]);
  auto vertices = point_cloud.getVertexPositions();
  auto colors = point_cloud.getVertexColors();

  const size_t data_size = vertices.size();
  glm::dvec3 *d_vertices_data;
  glm::ucvec3 *d_color_data;

  std::cout << "Points to render:" << data_size << std::endl;

  cudaError_t cudaStatus =
      cudaMalloc(&d_vertices_data, data_size * sizeof(glm::dvec3));
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaMalloc failed!" << cudaGetErrorString(cudaStatus)
              << std::endl;
    return 1;
  }

  cudaStatus = cudaMalloc(&d_color_data, data_size * sizeof(glm::ucvec3));
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaMalloc failed!" << cudaGetErrorString(cudaStatus)
              << std::endl;
    return 1;
  }

  std::cout << "Successful in allocating memory on the gpu" << std::endl;
  cudaMemcpy(d_vertices_data, vertices.data(), data_size * sizeof(glm::dvec3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_color_data, colors.data(), data_size * sizeof(glm::ucvec3),
             cudaMemcpyHostToDevice);

  std::cout << "Reading camera data" << std::endl;

  Camera cam;
  if (!readCameraFile(argv[2], cam)) {
    std::cerr << "Problem reading camera file: " << argv[2] << std::endl;
    return 1;
  }
  std::vector<glm::mat4> extrinsics;
  if (!readExtrinsics(argv[3], extrinsics)) {
    std::cerr << "Problem reading extrinsics file: " << argv[3] << std::endl;
    return 1;
  }

  uint8_t *d_output_image;

  cudaStatus = cudaMalloc(&d_output_image, cam.width * cam.height * 3);

  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaMalloc failed!" << cudaGetErrorString(cudaStatus)
              << std::endl;
    return 1;
  }

  double *d_depth_buffer;
  cudaStatus =
      cudaMalloc(&d_depth_buffer, cam.width * cam.height * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaMalloc failed!" << cudaGetErrorString(cudaStatus)
              << std::endl;
    return 1;
  }
  unsigned char *h_output_image =
      (unsigned char *)malloc(cam.height * cam.width * 3);

  int i = 0;
  for (auto &extrinsic : extrinsics) {
    cudaMemset(&d_depth_buffer, 100000,
               cam.width * cam.height * sizeof(double));

    glm::mat4 camProj = glm::mat4(glm::transpose(cam.intrinsic)) * extrinsic;

    glm::mat4 *d_cam_proj;
    cudaMalloc(&d_cam_proj, sizeof(glm::mat4));
    cudaMemcpy(d_cam_proj, glm::value_ptr(camProj), sizeof(glm::mat4),
               cudaMemcpyHostToDevice);

    int min_grid_size;
    int block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, naive);
    int num_blocks = (data_size + block_size - 1) / block_size;

    naive<<<num_blocks, block_size>>>(d_output_image, d_depth_buffer, cam.width,
                                      cam.height, d_cam_proj, d_vertices_data,
                                      d_color_data, data_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, cam.height * cam.width * 3,
               cudaMemcpyDeviceToHost);

    savePPMImage(std::string(argv[4]) + "/" + std::to_string(i++) + ".ppm",
                 h_output_image, cam.width, cam.height);
  }

  cudaFree(&d_vertices_data);
  cudaFree(&d_color_data);
  cudaFree(&d_depth_buffer);
  cudaFree(&d_output_image);
  free(h_output_image);

  return 0;
}