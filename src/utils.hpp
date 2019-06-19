// Utility functions

#pragma once

#include <string>
#include <vector>

void string_join(const std::vector<std::string>& strings, std::string separator, std::string& output) {
  output.clear();

  for (auto i = 0; i < strings.size(); i++) {
    output += strings[i];

    if (i != strings.size() - 1) {
      output += separator;
    }
  }
}
