#include "Timer.h"
#include "glm/fwd.hpp"
#include "glm/matrix.hpp"
#include "happly.h"
#include "render.cuh"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iomanip>
#include <limits>
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
                  int width, int height, bool binary = true,
                  bool verbose = false) {
  std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }

  // Write PPM header
  if (binary) {
    file << "P6\n";
  } else {
    file << "P3\n";
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
  if (verbose)
    std::cout << "Image saved to " << filename << "\r" << std::flush;
}

#define CUDA_ERROR(x)                                                          \
  if (x != cudaSuccess) {                                                      \
    std::cerr << "CUDA ERROR:" << cudaGetErrorString(x) << std::endl;          \
    return 1;                                                                  \
  }

struct Paths {
  std::string base;
  std::string data;
  std::string camera;
  std::string extrinsics;
  std::string output;
  bool succes = true;

  Paths(const std::string base_path, bool verbose = false) : base(base_path) {
    const std::filesystem::path base = base_path;
    const std::filesystem::path pc_aligned_rel = "scans/pc_aligned.ply";
    const std::filesystem::path cameras_rel = "iphone/colmap/cameras.txt";
    const std::filesystem::path images_rel = "iphone/colmap/images.txt";
    const std::filesystem::path renders_rel = "renders";

    data = base / pc_aligned_rel;
    camera = base / cameras_rel;
    extrinsics = base / images_rel;
    output = base / renders_rel;

    if (!std::filesystem::exists(data) ||
        !std::filesystem::is_regular_file(data)) {
      std::cerr << "Error: Required file not found or is not a regular file: "
                << data << std::endl;
      succes = false;
      return;
    }
    if (!std::filesystem::exists(camera) ||
        !std::filesystem::is_regular_file(camera)) {
      std::cerr << "Error: Required file not found or is not a regular file: "
                << camera << std::endl;
      succes = false;
      return;
    }
    if (!std::filesystem::exists(extrinsics) ||
        !std::filesystem::is_regular_file(extrinsics)) {
      std::cerr << "Error: Required file not found or is not a regular file: "
                << extrinsics << std::endl;
      succes = false;
      return;
    }
    if (!std::filesystem::exists(output) ||
        !std::filesystem::is_directory(output)) {
      std::cerr << "Error: Required directory not found or is not a directory, "
                   "creating directory: "
                << output << std::endl;
      std::filesystem::create_directory(output);
    }
    if (verbose)
      std::cout << "All required paths exist and are valid." << std::endl;
  }
};

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Wrong arguments suplied\nUsage: " << argv[0]
              << " <model_path>" << std::endl;
    return 1;
  }

  Paths paths(argv[1]);
  if (!paths.succes)
    return 1;

  std::vector<glm::dvec3> vertices;
  std::vector<glm::ucvec3> colors;

  {
    auto timer = ScopedTimer("Load PLY file");
    happly::PLYData point_cloud(paths.data);
    vertices = point_cloud.getVertexPositions();
    colors = point_cloud.getVertexColors();
  }

  const size_t data_size = vertices.size();
  glm::dvec3 *d_vertices_data;
  glm::ucvec3 *d_color_data;

  std::cout << "Points to in point cloud:" << data_size << std::endl;

  CUDA_ERROR(cudaMalloc(&d_vertices_data, data_size * sizeof(glm::dvec3)))
  CUDA_ERROR(cudaMalloc(&d_color_data, data_size * sizeof(glm::ucvec3)))

  CUDA_ERROR(cudaMemcpy(d_vertices_data, vertices.data(),
                        data_size * sizeof(glm::dvec3),
                        cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(d_color_data, colors.data(),
                        data_size * sizeof(glm::ucvec3),
                        cudaMemcpyHostToDevice));

  Camera cam;
  if (!readCameraFile(paths.camera, cam)) {
    std::cerr << "Problem reading camera file: " << paths.camera << std::endl;
    return 1;
  }
  std::vector<glm::mat4> extrinsics;
  if (!readExtrinsics(paths.extrinsics, extrinsics)) {
    std::cerr << "Problem reading extrinsics file: " << paths.extrinsics
              << std::endl;
    return 1;
  }
  std::cout << "Rendering " << extrinsics.size() << " images" << std::endl;

  uint8_t *d_output_image;
  double *d_depth_buffer;
  glm::mat4 *d_cam_proj;
  CUDA_ERROR(cudaMalloc(&d_output_image, cam.width * cam.height * 3));
  CUDA_ERROR(
      cudaMalloc(&d_depth_buffer, cam.width * cam.height * sizeof(double)))
  CUDA_ERROR(cudaMalloc(&d_cam_proj, sizeof(glm::mat4)));
  unsigned char *h_output_image =
      (unsigned char *)malloc(cam.height * cam.width * 3);

  int min_grid_size;
  int block_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, naive);
  int num_blocks = (data_size + block_size - 1) / block_size;

  int i = 0;
  for (auto &extrinsic : extrinsics) {
    fillDoubleArrayKernel<<<num_blocks, block_size>>>(
        d_depth_buffer, std::numeric_limits<double>::max(),
        cam.width * cam.height);

    glm::mat4 camProj = glm::mat4(glm::transpose(cam.intrinsic)) * extrinsic;

    CUDA_ERROR(cudaMemcpy(d_cam_proj, glm::value_ptr(camProj),
                          sizeof(glm::mat4), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemset(d_output_image, 0, cam.width * cam.height * 3));
    memset(h_output_image, 0, cam.width * cam.height * 3);

    naive<<<num_blocks, block_size>>>(d_output_image, d_depth_buffer, cam.width,
                                      cam.height, d_cam_proj, d_vertices_data,
                                      d_color_data, data_size);

    cudaDeviceSynchronize();

    CUDA_ERROR(cudaMemcpy(h_output_image, d_output_image,
                          cam.height * cam.width * 3, cudaMemcpyDeviceToHost));

    savePPMImage(paths.output + "/" + std::to_string(i++) + ".ppm",
                 h_output_image, cam.width, cam.height, true, true);
  }

  cudaFree(&d_vertices_data);
  cudaFree(&d_color_data);
  cudaFree(&d_depth_buffer);
  cudaFree(&d_output_image);
  free(h_output_image);

  return 0;
}