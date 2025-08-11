#include <sys/types.h>
#include <thrust/reduce.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iomanip>
#include <sstream>
#include <string>

#include "AsyncImageWriter.h"
#include "Timer.h"
#include "glm/fwd.hpp"
#include "glm/matrix.hpp"
#include "happly.h"
#include "render.cuh"
#include "types.h"

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
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);

    std::string model;
    float fx, fy, cx, cy;
    float k1, k2, p1, p2;

    iss >> cam.id >> model >> cam.width >> cam.height >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2;

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
}

// Function to print a glm::mat4
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

bool readExtrinsics(const std::string &file_path, std::vector<glm::mat4> &extrinsics, bool single = false) {
  std::ifstream configurationFile(file_path);
  if (!configurationFile.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(configurationFile, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);

    glm::quat rotation;
    glm::vec3 translation;
    uint id;

    iss >> id >> rotation.w >> rotation.x >> rotation.y >> rotation.z >> translation.x >> translation.y >>
        translation.z;

    glm::mat4 extrinsic = glm::mat4_cast(rotation);
    extrinsic[3][0] = translation.x;
    extrinsic[3][1] = translation.y;
    extrinsic[3][2] = translation.z;

    extrinsics.push_back(extrinsic);
    if (single) break;
  }

  return true;
}
// Function to save raw RGBA pixel data to a PPM image file
void savePPMImage(const std::string &filename, const uchar3 *data, int width, int height, bool binary = true,
                  bool verbose = false) {
  std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
    return;
  }

  // Write PPM header
  if (binary) {
    file << "P6\n";
  } else {
    file << "P3\n";
  }

  file << width << " " << height << "\n";
  file << "255\n";  // Max color value

  // Write pixel data (only RGB, skip Alpha for PPM P6)
  for (int i = 0; i < width * height; ++i) {
    if (binary) {
      file.write(reinterpret_cast<const char *>(&data[i].x), 1);  // Red
      file.write(reinterpret_cast<const char *>(&data[i].y), 1);  // Green
      file.write(reinterpret_cast<const char *>(&data[i].z), 1);  // Blue
    } else {
      file << std::to_string(data[i].x) << " " << std::to_string(data[i].y) << " " << std::to_string(data[i].z) << "\n";
    }
  }

  file.close();
  if (verbose) std::cout << "Image saved to " << filename << "\r" << std::flush;
}

#define CUDA_ERROR(x)                                                                      \
  if (x != cudaSuccess) {                                                                  \
    std::cerr << "CUDA ERROR (" << __LINE__ << "):" << cudaGetErrorString(x) << std::endl; \
    return 1;                                                                              \
  }

struct Paths {
  std::string base;
  std::string data;
  std::string camera;
  std::string extrinsics;
  std::string output;
  std::string depth;
  bool succes = true;

  Paths(const std::string base_path, bool verbose = false) : base(base_path) {
    const std::filesystem::path base = base_path;
    const std::filesystem::path pc_aligned_rel = "scans/pc_aligned.ply";
    const std::filesystem::path cameras_rel = "iphone/colmap/cameras.txt";
    const std::filesystem::path images_rel = "iphone/colmap/images.txt";
    const std::filesystem::path renders_rel = "renders";
    const std::filesystem::path depth_rel = "depth";

    data = base / pc_aligned_rel;
    camera = base / cameras_rel;
    extrinsics = base / images_rel;
    output = base / renders_rel;
    depth = base / depth_rel;

    if (!std::filesystem::exists(data) || !std::filesystem::is_regular_file(data)) {
      std::cerr << "Error: Required file not found or is not a regular file: " << data << std::endl;
      succes = false;
      return;
    }
    if (!std::filesystem::exists(camera) || !std::filesystem::is_regular_file(camera)) {
      std::cerr << "Error: Required file not found or is not a regular file: " << camera << std::endl;
      succes = false;
      return;
    }
    if (!std::filesystem::exists(extrinsics) || !std::filesystem::is_regular_file(extrinsics)) {
      std::cerr << "Error: Required file not found or is not a regular file: " << extrinsics << std::endl;
      succes = false;
      return;
    }
    if (!std::filesystem::exists(output) || !std::filesystem::is_directory(output)) {
      std::cerr << "Error: Required directory not found or is not a directory, "
                   "creating directory: "
                << output << std::endl;
      std::filesystem::create_directory(output);
    }
    if (!std::filesystem::exists(depth) || !std::filesystem::is_directory(depth)) {
      std::cerr << "Error: Required directory not found or is not a directory, "
                   "creating directory: "
                << depth << std::endl;
      std::filesystem::create_directory(depth);
    }
    if (verbose) std::cout << "All required paths exist and are valid." << std::endl;
  }
};

size_t calculateDynamicSharedMemory(int blockSize) { return blockSize * sizeof(PCData); }

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Wrong arguments suplied\nUsage: " << argv[0] << " <model_path>" << std::endl;
    return 1;
  }

  Paths paths(argv[1]);
  if (!paths.succes) return 1;

  // Read point cloud data
  std::vector<float4> vertices;
  std::vector<uchar4> colors;

  happly::PLYData point_cloud(paths.data);
  vertices = point_cloud.getVertexPositions();
  colors = point_cloud.getVertexColors();

  std::cout << "Points to in point cloud: " << vertices.size() << std::endl;

  Camera cam;
  if (!readCameraFile(paths.camera, cam)) {
    std::cerr << "Problem reading camera file: " << paths.camera << std::endl;
    return 1;
  }
  std::vector<glm::mat4> extrinsics;
  if (!readExtrinsics(paths.extrinsics, extrinsics, false)) {
    std::cerr << "Problem reading extrinsics file: " << paths.extrinsics << std::endl;
    return 1;
  }
  std::cout << "Rendering " << extrinsics.size() << " images" << std::endl;

  // Device buffers
  float4 *d_vertices_data;
  uchar4 *d_color_data;
  uint64_t *d_output;
  glm::ucvec3 *d_image;
  glm::ucvec3 *d_depth;
  glm::mat4 *d_cam_proj;
  uint32_t *d_block_maxes;
  uint32_t *d_absolute_max;

  // Host buffers
  uchar3 *h_image;
  cudaHostAlloc((void **)&h_image, cam.width * cam.height * sizeof(uchar3), cudaHostAllocDefault);
  uchar3 *h_depth;
  cudaHostAlloc((void **)&h_depth, cam.width * cam.height * sizeof(uchar3), cudaHostAllocDefault);
  const size_t data_size = vertices.size();
  uint2 image_size = {cam.width, cam.height};

  CUDA_ERROR(cudaMalloc(&d_vertices_data, data_size * sizeof(float4)));
  CUDA_ERROR(cudaMalloc(&d_color_data, data_size * sizeof(uint4)));
  CUDA_ERROR(cudaMalloc(&d_output, cam.width * cam.height * sizeof(uint64_t)));
  CUDA_ERROR(cudaMalloc(&d_image, cam.width * cam.height * sizeof(glm::ucvec3)));
  CUDA_ERROR(cudaMalloc(&d_depth, cam.width * cam.height * sizeof(glm::ucvec3)));
  CUDA_ERROR(cudaMalloc(&d_cam_proj, sizeof(glm::mat4)));
  CUDA_ERROR(cudaMalloc(&d_block_maxes, cam.width * cam.height * sizeof(uint32_t)));
  CUDA_ERROR(cudaMalloc(&d_absolute_max, sizeof(uint32_t)))

  CUDA_ERROR(cudaMemcpy(d_vertices_data, vertices.data(), data_size * sizeof(float4), cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(d_color_data, colors.data(), data_size * sizeof(uchar4), cudaMemcpyHostToDevice));

  AsyncImageWriter image_writer(std::thread::hardware_concurrency() - 1);

  int min_grid_size;
  int block_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, vertexOrderOptimization);
  int num_blocks = (data_size + block_size - 1) / block_size;

  dim3 block_dim(16, 16);
  dim3 grid_dim(cam.width / block_dim.x, cam.height / block_dim.y);

  int i = 0;
  for (auto &extrinsic : extrinsics) {
    fillBuffer<<<grid_dim, block_dim>>>(d_output, 0xFFFFFFFFFF000000, cam.width * cam.height);

    glm::mat4 camProj = glm::transpose(glm::mat4(glm::transpose(cam.intrinsic)) * extrinsic);

    CUDA_ERROR(cudaMemcpy(d_cam_proj, glm::value_ptr(camProj), sizeof(glm::mat4), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemset(h_image, 0, cam.width * cam.height * sizeof(uchar3)))

    {
      auto timer = ScopedTimer("VOO optim " + std::to_string(i));
      vertexOrderOptimization<<<num_blocks, block_size>>>(d_output, d_vertices_data, d_color_data, (float *)d_cam_proj,
                                                          image_size, data_size);
      cudaDeviceSynchronize();
      findBlockMaxKernel<<<grid_dim, block_dim, 16 * 16 * sizeof(uint32_t)>>>(d_output, d_block_maxes, data_size);
      cudaDeviceSynchronize();
      findAbsoluteMaxKernel<<<1, block_dim>>>(d_block_maxes, d_absolute_max, grid_dim.x * grid_dim.y);
      cudaDeviceSynchronize();
      resolve<<<grid_dim, block_dim>>>(d_image, d_depth, d_absolute_max, d_output, cam.width * cam.height);
      cudaDeviceSynchronize();
    }

    CUDA_ERROR(cudaMemcpy(h_image, d_image, cam.height * cam.width * sizeof(uchar3), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(h_depth, d_depth, cam.height * cam.width * sizeof(uchar3), cudaMemcpyDeviceToHost));

    ImageSaveTask current_task;
    current_task.filename = paths.output + "/" + std::to_string(i) + ".ppm";
    current_task.pixel_data.assign(h_image, h_image + cam.width * cam.height);
    current_task.width = cam.width;
    current_task.height = cam.height;
    image_writer.enqueue(std::move(current_task));

    current_task.filename = paths.depth + "/" + std::to_string(i) + ".ppm";
    current_task.pixel_data.assign(h_depth, h_depth + cam.width * cam.height);
    current_task.width = cam.width;
    current_task.height = cam.height;
    image_writer.enqueue(std::move(current_task));

    i++;
  }

  cudaFree(d_vertices_data);
  cudaFree(d_color_data);
  cudaFree(d_output);
  cudaFree(d_image);
  cudaFree(d_cam_proj);
  cudaFree(d_block_maxes);
  cudaFree(d_absolute_max);
  cudaFreeHost(h_image);

  // std::string command = "ffmpeg -y -framerate 6 -i " + paths.output + "/%d.ppm -c:v libx264 -pix_fmt yuv420p -r 6 " +
  //                       paths.output + "/output.mp4";

  // int result = system(command.c_str());
  // if (result != 0) std::cerr << "Video conversion failed with error code: " << result << std::endl;

  return 0;
}