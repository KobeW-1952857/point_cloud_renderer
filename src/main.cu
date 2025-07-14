#include "Timer.h"
#include "glm/fwd.hpp"
#include "glm/matrix.hpp"
#include "happly.h"
#include "render.cuh"
#include "lod/lod.cuh"
#include "lod/lod_util.cuh"

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
#include <utility>
#include <unordered_map>

DVecPair findAABB(glm::dvec3* vertices, size_t vertexAmt) {
  glm::dvec3 bmin(std::numeric_limits<double>::max());
  glm::dvec3 bmax(std::numeric_limits<double>::lowest());

  for (size_t i = 0; i < vertexAmt; ++i) {
    const auto& v = vertices[i];
    bmin = glm::min(bmin, v);
    bmax = glm::max(bmax, v);
  }

  return { bmin, bmax };
}

void buildLODStructure(glm::dvec3* pos_host, glm::dvec3* pos_device, glm::ucvec3* col, size_t vertexAmt,
                       unsigned int* outputCounter, CLODPoints output) {
  auto aabb = findAABB(pos_host, vertexAmt);

  buildCLODLevels_denseGrid(pos_device, col, vertexAmt, aabb.first, aabb.second,
                             ROOT_SPACING, LEVELS_AMT, output, outputCounter);
}

void printLevelHistogram(size_t vertexAmt, const CLODPoints* d_output) {
  std::vector<glm::ucvec4> host_cols(vertexAmt);
  if (cudaMemcpy(host_cols.data(), d_output->cols, vertexAmt * sizeof(glm::ucvec4), cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "CUDA memcpy failed" << std::endl;
    return;
  }

  std::unordered_map<int, size_t> level_counts;
  for (const auto& col : host_cols) {
    level_counts[col.w]++;
  }

  std::cout << "Points per level:\n";
  for (const auto& [level, count] : level_counts) {
    std::cout << "  Level " << level << ": " << count << " points\n";
  }
}


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

void render(const std::vector<glm::mat4>& extrinsics,
            const Camera& cam,
            const std::string& output_path,
            glm::dvec3* d_vertices_data,
            glm::ucvec3* d_color_data,
            size_t data_size,
            uint8_t* d_output_image,
            double* d_depth_buffer,
            glm::mat4* d_cam_proj,
            unsigned char* h_output_image,
            bool enable_timing) {
  
  int min_grid_size, block_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, naive);
  int num_blocks = (data_size + block_size - 1) / block_size;

  OptionalTimerWriter timer_writer(enable_timing, "../timings_naive.txt");

  for (size_t i = 0; i < extrinsics.size(); ++i) {
    const auto& extrinsic = extrinsics[i];
    SteadyTimer timer;

    fillDoubleArrayKernel<<<num_blocks, block_size>>>(
        d_depth_buffer,
        std::numeric_limits<double>::max(),
        cam.width * cam.height
    );

    glm::mat4 cam_proj = glm::mat4(glm::transpose(cam.intrinsic)) * extrinsic;
    cudaMemcpy(d_cam_proj, glm::value_ptr(cam_proj),
               sizeof(glm::mat4), cudaMemcpyHostToDevice);

    cudaMemset(d_output_image, 0, cam.width * cam.height * 3);
    memset(h_output_image, 0, cam.width * cam.height * 3);

    // naive<<<num_blocks, block_size>>>(
    //     d_output_image, d_depth_buffer,
    //     cam.width, cam.height,
    //     d_cam_proj,
    //     d_vertices_data, d_color_data, data_size
    // );

    cudaDeviceSynchronize();
    timer_writer.write(timer.ElapsedMillis());

    cudaMemcpy(h_output_image, d_output_image,
               cam.width * cam.height * 3, cudaMemcpyDeviceToHost);
    savePPMImage(output_path + "/" + std::to_string(i) + ".ppm",
                 h_output_image, cam.width, cam.height, true, true);
  }
}


void renderLODs(const std::vector<glm::mat4>& extrinsics,
                const Camera& cam,
                const std::string& output_path,
                CLODPoints clod_points,
                size_t data_size,
                uint8_t* d_output_image,
                double* d_depth_buffer,
                glm::mat4* d_cam_proj,
                unsigned char* h_output_image,
                glm::dvec3* pos_render_buf,
                glm::ucvec3* col_render_buf,
                bool enable_timing) {
  
  unsigned int* d_filtered_points_amt = nullptr;
  cudaMalloc(&d_filtered_points_amt, sizeof(unsigned int));

  int min_grid_size, block_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, naive);
  int num_blocks = (data_size + block_size - 1) / block_size;
  
  OptionalTimerWriter timer_writer(enable_timing, "../timings_lod.txt");

  for (size_t i = 0; i < extrinsics.size(); ++i) {
    const auto& extrinsic = extrinsics[i];
    SteadyTimer timer;

    cudaMemset(d_filtered_points_amt, 0, sizeof(unsigned int));

    fillDoubleArrayKernel<<<num_blocks, block_size>>>(
        d_depth_buffer,
        std::numeric_limits<double>::max(),
        cam.width * cam.height
    );

    glm::vec3 cam_pos = glm::vec3(glm::inverse(extrinsic)[3]);
    glm::mat4 cam_proj = glm::mat4(glm::transpose(cam.intrinsic)) * extrinsic;
    cudaMemcpy(d_cam_proj, glm::value_ptr(cam_proj),
               sizeof(glm::mat4), cudaMemcpyHostToDevice);

    cudaMemset(d_output_image, 0, cam.width * cam.height * 3);
    memset(h_output_image, 0, cam.width * cam.height * 3);

    filterPointsKernel<<<num_blocks, block_size>>>(
        clod_points, cam_pos,
        ROOT_SPACING, CLOD_FACTOR,
        data_size,
        pos_render_buf, col_render_buf,
        d_filtered_points_amt
    );
    cudaDeviceSynchronize();

    naive<<<num_blocks, block_size>>>(
        d_output_image, d_depth_buffer,
        cam.width, cam.height,
        d_cam_proj,
        pos_render_buf, col_render_buf, d_filtered_points_amt
    );
    cudaDeviceSynchronize();

    timer_writer.write(timer.ElapsedMillis());

    cudaMemcpy(h_output_image, d_output_image,
               cam.width * cam.height * 3, cudaMemcpyDeviceToHost);
    savePPMImage(output_path + "/" + std::to_string(i) + ".ppm",
                 h_output_image, cam.width, cam.height, true, true);
  }

  cudaFree(d_filtered_points_amt);
}



int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path>\n";
    return 1;
  }

  Paths paths(argv[1]);
  if (!paths.succes) return 1;

  bool debug = false;
  if (argc == 3) {
    if (std::strcmp(argv[2], "--debug") == 0) {
      debug = true;
    } else {
      std::cerr << "Usage: " << argv[0] << " <model_path> " << "(--debug)\n";
      return 1;
    }
  }

  std::vector<glm::dvec3> vertices;
  std::vector<glm::ucvec3> colors;

  {
    auto timer = ScopedTimer("Load PLY file");
    happly::PLYData ply(paths.data);
    vertices = ply.getVertexPositions();
    colors = ply.getVertexColors();
  }

  size_t data_size = vertices.size();
  std::cout << "Loaded " << data_size << " points.\n";

  glm::dvec3* d_vertices_data;
  glm::ucvec3* d_color_data;
  CUDA_ERROR(cudaMalloc(&d_vertices_data, data_size * sizeof(glm::dvec3)));
  CUDA_ERROR(cudaMalloc(&d_color_data, data_size * sizeof(glm::ucvec3)));
  CUDA_ERROR(cudaMemcpy(d_vertices_data, vertices.data(), data_size * sizeof(glm::dvec3), cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(d_color_data, colors.data(), data_size * sizeof(glm::ucvec3), cudaMemcpyHostToDevice));

  Camera cam;
  if (!readCameraFile(paths.camera, cam)) {
    std::cerr << "Failed to load camera file.\n";
    return 1;
  }

  std::vector<glm::mat4> extrinsics;
  if (!readExtrinsics(paths.extrinsics, extrinsics)) {
    std::cerr << "Failed to load extrinsics file.\n";
    return 1;
  }

  std::cout << "Rendering " << extrinsics.size() << " frames...\n";

  uint8_t* d_output_image;
  double* d_depth_buffer;
  glm::mat4* d_cam_proj;
  CUDA_ERROR(cudaMalloc(&d_output_image, cam.width * cam.height * 3));
  CUDA_ERROR(cudaMalloc(&d_depth_buffer, cam.width * cam.height * sizeof(double)));
  CUDA_ERROR(cudaMalloc(&d_cam_proj, sizeof(glm::mat4)));

  auto* h_output_image = (unsigned char*)malloc(cam.width * cam.height * 3);

  // LOD
  CLODPoints clodPoints;
  unsigned int* clodCounter;
  CUDA_ERROR(cudaMalloc(&clodCounter, sizeof(unsigned int)));
  CUDA_ERROR(cudaMemset(clodCounter, 0, sizeof(unsigned int)));
  CUDA_ERROR(cudaMalloc(&clodPoints.positions, data_size * sizeof(glm::dvec3)));
  CUDA_ERROR(cudaMalloc(&clodPoints.cols, data_size * sizeof(glm::ucvec4)));

  // final rendering buffer
  glm::dvec3* pos_render_buffer;
  glm::ucvec3* color_render_buffer;
  CUDA_ERROR(cudaMalloc(&pos_render_buffer, data_size * sizeof(glm::dvec3)));
  CUDA_ERROR(cudaMalloc(&color_render_buffer, data_size * sizeof(glm::ucvec3)));

  buildLODStructure(vertices.data(), d_vertices_data, d_color_data, data_size, clodCounter, clodPoints);
  printLevelHistogram(data_size, &clodPoints);

  renderLODs(extrinsics, cam, paths.output, clodPoints, data_size,
             d_output_image, d_depth_buffer, d_cam_proj, h_output_image,
             pos_render_buffer, color_render_buffer, debug);

  // render(extrinsics, cam, paths.output, d_vertices_data, d_color_data, 
  //   data_size, d_output_image, d_depth_buffer, d_cam_proj, h_output_image, debug);

  // cleanup
  cudaFree(d_vertices_data);
  cudaFree(d_color_data);
  cudaFree(d_output_image);
  cudaFree(d_depth_buffer);
  cudaFree(d_cam_proj);
  cudaFree(clodPoints.positions);
  cudaFree(clodPoints.cols);
  cudaFree(pos_render_buffer);
  cudaFree(color_render_buffer);
  free(h_output_image);

  return 0;
}
