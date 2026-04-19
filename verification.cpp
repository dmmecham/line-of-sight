#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

bool areFilesEqual(const std::string& filePath1, const std::string& filePath2) {
  std::ifstream file1(filePath1, std::ios::binary | std::ios::ate);
  std::ifstream file2(filePath2, std::ios::binary | std::ios::ate);

  // Make sure the files can be accessed.
  if (!file1.is_open() || !file2.is_open()) {
    std::cerr << "Error opening files.\n";
    return false;
  }

  // Get the file size from the end point.
  size_t file1Size = file1.tellg();
  size_t file2Size = file2.tellg();

  // Compare file sizes as a shortcut.
  if (file1Size != file2Size) {
    return false;
  }

  // Seek back to the beginning to do the proper check.
  std::streamsize size = file1.tellg();
  file1.seekg(0);
  file2.seekg(0);

  // File contents.
  std::vector<uint8_t> file1Data(file1Size);
  if (!file1.read(reinterpret_cast<char*>(file1Data.data()), file1Size)) {
    throw std::runtime_error("Error reading input file 1.");
  }
  std::vector<uint8_t> file2Data(file2Size);
  if (!file2.read(reinterpret_cast<char*>(file2Data.data()), file2Size)) {
    throw std::runtime_error("Error reading input file 2.");
  }

  return std::memcmp(file1Data.data(), file2Data.data(), file1Size);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <file1> <file2>\n";
    return 1;
  }

  try {
    if (areFilesEqual(argv[1], argv[2])) {
      std::cout << "Files are identical" << std::endl;
    } else {
      std::cout << "Files do not match" << std::endl;
      return 1;
    }
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
