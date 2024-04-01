#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <cassert>
#include <stdexcept>
#include <numeric>


// TODO: Include necessary C++ libraries for data manipulation (e.g., DataFrames)

// class EntrySampler {
// public:
//     EntrySampler(const std::string& path) {
//         // TODO: Implement the constructor logic
//     }

//     int sample(const std::string& date) {
//         // TODO: Implement the sampling logic
//         return 0;
//     }

// private:
//     // TODO: Define necessary member variables
// };

// class ToyEntrySampler {
// public:
//     ToyEntrySampler() {}

//     int sample(double mean = 60.0, double sd = 20.0) {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::normal_distribution<> dis(mean, sd);
//         return static_cast<int>(std::round(std::max(0.0, dis(gen))));
//     }
// };

class QuickEntrySampler {
public:
    QuickEntrySampler(const std::string& numEntryPath) {
        std::ifstream file(numEntryPath);
        if (file.is_open()) {
            std::string line;
            std::getline(file, line); // Skip the header line

            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string field;

                std::getline(ss, field, ','); // index
                std::getline(ss, field, ','); // date
                std::getline(ss, field, ','); // num_entry
                int numEntries = std::stoi(field);
                std::getline(ss, field, ','); // weekday
                std::string weekday = field;

                if (weekday == "Saturday" || weekday == "Sunday") {
                    weekendEntries.push_back(numEntries);
                } else {
                    weekdayEntries.push_back(numEntries);
                }
            }
            file.close();
        }
    }

    int sample(const std::string& date) {
        std::string weekday = getWeekday(date);

        if (weekday == "Saturday" || weekday == "Sunday") {
            return sampleFromVector(weekendEntries);
        } else {
            return sampleFromVector(weekdayEntries);
        }
    }

private:
    std::vector<int> weekendEntries;
    std::vector<int> weekdayEntries;

    int sampleFromVector(const std::vector<int>& entries) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, entries.size() - 1);
        return entries[dis(gen)];
    }

    std::string getWeekday(const std::string& dateStr) {
        std::tm tm = {};
        std::istringstream ss(dateStr);
        ss >> std::get_time(&tm, "%Y-%m-%d");

        if (ss.fail()) {
            // Handle invalid date format
            throw std::runtime_error("Invalid date format");
        }

        std::time_t time = std::mktime(&tm);
        std::tm* ltm = std::localtime(&time);

        std::string weekdays[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
        return weekdays[ltm->tm_wday];
    }
};


// class DurationSampler {
// public:
//     DurationSampler(const std::string& path) {
//         // TODO: Implement the constructor logic
//     }

//     int sample() {
//         // TODO: Implement the sampling logic
//         return 0;
//     }

// private:
//     // TODO: Define necessary member variables
// };

// class ToyDurationSampler {
// public:
//     ToyDurationSampler() {}

//     int sample(double mean = 1.3, double sd = 0.95) {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::lognormal_distribution<> dis(mean, sd);
//         return static_cast<int>(std::round(std::max(0.0, dis(gen))));
//     }
// };

class QuickDurationSampler {
public:
    QuickDurationSampler(const std::string& durationPath) {
        std::ifstream file(durationPath);
        if (file.is_open()) {
            std::string line;
            std::getline(file, line); // Skip the header line

            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string field1;
                std::string field2;

                std::getline(ss, field1, ','); // ID
                std::getline(ss, field2, ','); // Duration
                int duration = std::stoi(field2);

                durations.push_back(duration);
            }
            file.close();
        }
    }

    int sample() {
        return sampleFromVector(durations);
    }

private:
    std::vector<int> durations;

    int sampleFromVector(const std::vector<int>& entries) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, entries.size() - 1);
        return entries[dis(gen)];
    }
};

class PathSampler {

public:
    PathSampler(const std::string& matrix_path = "") { 
        transition_matrix_folder_path = matrix_path;
    }

    std::vector<std::string> sample(int duration, const std::string& method, int window_size = 0) {
        assert(duration >= 0);

        std::string filename = transition_matrix_folder_path + "/" + std::to_string(duration) + "_day_";
        if (method == "sliding_window") {
            filename += "sw_" + std::to_string(window_size);
        } else if (method == "shorter_only") {
            filename += "so";
        } else if (method == "longer_only") {
            filename += "lo";
        }
        filename += ".csv";

        load_transition_matrix(filename);

        std::vector<std::string> path;
        std::string current_state = "ADMISSION";
        std::string end_state = "DISCHARGE";

        path.push_back(current_state);
        if (duration > 0) {
            for (int i = 0; i < duration; ++i) {
                current_state = pick_next_state(current_state);
                path.push_back(current_state);
            }
        } else {
            // for testing only
            // while (current_state != end_state) {
            //     current_state = pick_next_state(current_state, end_state);
            //     path.push_back(current_state);
            // }
        }
        path.push_back(end_state);
        return path;
    }

private:
    std::unordered_map<std::string, std::unordered_map<std::string, double>> transition_matrix;
    std::string transition_matrix_folder_path;
    int num_states;

    void load_transition_matrix(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;
        std::vector<std::string> states;
        int i = -1; // row index

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;

            std::getline(ss, cell, ','); // skip the first column
            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }

            if (i == -1) {
                states = row; // This is the header row
                i += 1;
                continue;
            }

            for (size_t j = 0; j < row.size(); j++) {
                transition_matrix[states[i]][states[j]] = std::stod(row[j]);
            }

            i += 1;
        }
    }

    std::string pick_next_state(const std::string& current_state) {
        auto& probabilities = transition_matrix[current_state];

        // Extract the weights for the distribution
        std::vector<double> weights;
        for (const auto& pair : probabilities) {
            weights.push_back(pair.second);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(weights.begin(), weights.end());

        // Generate a random index based on the distribution and find the corresponding next state
        auto it = std::next(probabilities.begin(), dist(gen));
        return it->first;
    }
};


// class ToyPathSampler {
// public:
//     ToyPathSampler(int numDepartments = 10) {
//         // TODO: Implement the constructor logic
//     }

//     std::vector<std::string> sample(int duration) {
//         // TODO: Implement the sampling logic
//         return {};
//     }

// private:
//     // TODO: Define necessary member variables
// };

std::unordered_map<std::string, std::vector<std::vector<std::string> > > generateMovement(
    const std::string& transitionMatrixPath,
    const std::string& startDateStr,
    const std::string& endDateStr,
    const std::string& method,
    int windowSize = 0,
    const std::string& dataPath = "",
    bool toy = false,
    int numToyDepartments = 10,
    bool quick = false,
    const std::string& numEntryPath = "",
    const std::string& durationPath = ""
) {
    //TODO:
    QuickEntrySampler* entrySampler;
    QuickDurationSampler* durationSampler;
    PathSampler* pathSampler;

    if (toy) {
        // entrySampler = new ToyEntrySampler();
        // durationSampler = new ToyDurationSampler();
        // pathSampler = new ToyPathSampler(numToyDepartments);
    } else if (quick) {
        entrySampler = new QuickEntrySampler(numEntryPath);
        durationSampler = new QuickDurationSampler(durationPath);
        pathSampler = new PathSampler(transitionMatrixPath);
    } else {
        // entrySampler = new EntrySampler(dataPath);
        // durationSampler = new DurationSampler(dataPath);
        // pathSampler = new PathSampler(transitionMatrixPath);
    }

    std::unordered_map<std::string, std::vector<std::vector<std::string> > > dailyPaths;

    // Step 1: Determine Start and End Date
    std::tm startTm = {};
    std::istringstream startIss(startDateStr);
    startIss >> std::get_time(&startTm, "%Y-%m-%d");
    std::chrono::system_clock::time_point startDate = std::chrono::system_clock::from_time_t(std::mktime(&startTm));

    std::tm endTm = {};
    std::istringstream endIss(endDateStr);
    endIss >> std::get_time(&endTm, "%Y-%m-%d");
    std::chrono::system_clock::time_point endDate = std::chrono::system_clock::from_time_t(std::mktime(&endTm));

    // Step 2: For each day sample number of entry
    auto currentDate = startDate;
    while (currentDate <= endDate) {
        std::time_t currentTime = std::chrono::system_clock::to_time_t(currentDate);
        std::tm* currentTm = std::localtime(&currentTime);
        char dateStr[11];
        std::strftime(dateStr, sizeof(dateStr), "%Y-%m-%d", currentTm);

        int numEntries;
        if (toy) {
            // numEntries = entrySampler->sample();
        } else {
            numEntries = entrySampler->sample(dateStr);
        }

        std::vector<std::vector<std::string> > oneDayPaths;
        for (int i = 0; i < numEntries; ++i) {
            // Step 3: For each entry sample length of duration
            int duration;
            if (toy) {
                duration = durationSampler->sample();
            } else {
                duration = durationSampler->sample();
            }
            if (duration == 0) {
                duration = 1;
            }

            // Step 4: For each duration sample path
            std::vector<std::string> path;
            if (toy) {
                // path = pathSampler->sample(duration);
            } else {
                path = pathSampler->sample(duration, method, windowSize);
            }

            oneDayPaths.push_back(path);
        }

        dailyPaths[dateStr] = oneDayPaths;
        currentDate += std::chrono::hours(24);
    }

    delete entrySampler;
    delete durationSampler;
    delete pathSampler;

    return dailyPaths;
}

std::string formatDate(const std::string& dateStr, int days) {
    std::istringstream iss(dateStr);
    std::tm tm = {};
    iss >> std::get_time(&tm, "%Y-%m-%d");

    tm.tm_mday += days;
    std::mktime(&tm);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d");
    return oss.str();
}

std::vector<std::unordered_map<std::string, std::string> > pathToMovement(const std::unordered_map<std::string, std::vector<std::vector<std::string> > >& pathData) {
    std::vector<std::unordered_map<std::string, std::string> > movementData;
    int patientIdCounter = 1;

    for (const auto& entry : pathData) {
        const std::string& startDateStr = entry.first;
        const auto& paths = entry.second;

        for (const auto& path : paths) {
            std::string patientId = "A_" + std::to_string(patientIdCounter);
            patientIdCounter++;

            for (size_t i = 0; i < path.size() - 1; ++i) {
                const std::string& fromDept = path[i];
                const std::string& toDept = path[i + 1];
                std::string stepDateStr = formatDate(startDateStr, i);

                std::unordered_map<std::string, std::string> movementEntry;
                movementEntry["id"] = patientId;
                movementEntry["date"] = stepDateStr;
                movementEntry["from_department"] = fromDept;
                movementEntry["to_department"] = toDept;

                movementData.push_back(movementEntry);
            }
        }
    }

    // Sort the movement data by date
    std::sort(movementData.begin(), movementData.end(), [](const std::unordered_map<std::string, std::string>& a, const std::unordered_map<std::string, std::string>& b) {
        return a.at("date") < b.at("date");
    });

    return movementData;
}


void runGeneration(
    int numSample,
    const std::string& transitionMatrixFolderPath,
    const std::string& outputFolderPath,
    const std::string& startDateStr,
    const std::string& endDateStr,
    const std::string& method,
    int windowSize = 0,
    const std::string& dataPath = "",
    bool toy = false,
    int numToyDepartments = 10,
    bool quick = false,
    const std::string& numEntryPath = "",
    const std::string& durationPath = ""
) {
    for (int i = 0; i < numSample; ++i) {
        auto dailyPaths = generateMovement(
            transitionMatrixFolderPath,
            startDateStr,
            endDateStr,
            method,
            windowSize,
            dataPath,
            toy,
            numToyDepartments,
            quick,
            numEntryPath,
            durationPath
        );

        auto generatedMovement = pathToMovement(dailyPaths);

        int fileIndex = 1;
        while (true) {
            std::string fileName;
            if (toy) {
                fileName = "toy_" + std::to_string(fileIndex) + ".csv";
            } else if (method == "sliding_window") {
                fileName = "sw_" + std::to_string(windowSize) + "_" + std::to_string(fileIndex) + ".csv";
            } else if (method == "shorter_only") {
                fileName = "so_" + std::to_string(fileIndex) + ".csv";
            } else if (method == "longer_only") {
                fileName = "lo_" + std::to_string(fileIndex) + ".csv";
            }

            std::string filePath = outputFolderPath + "/" + fileName;
            if (!std::filesystem::exists(filePath)) {
                std::ofstream outputFile(filePath);
                if (outputFile.is_open()) {
                    // Write header
                    outputFile << "id,date,from_department,to_department\n";

                    // Write movement data
                    for (const auto& movement : generatedMovement) {
                        outputFile << movement.at("id") << ","
                                   << movement.at("date") << ","
                                   << movement.at("from_department") << ","
                                   << movement.at("to_department") << "\n";
                    }

                    outputFile.close();
                    std::cout << "DataFrame saved to " << filePath << std::endl;
                    break;
                }
            }
            fileIndex++;
        }
    }
}

int main() {
    std::string transitionMatrixFolderPath = "/Users/hz9/dev/HAI_simulation/movement_generation/deid_data/transition_matrices/";
    std::string outputFolderPath = "/Users/hz9/dev/HAI_simulation/movement_generation/output/";
    std::string startDateStr = "2024-01-01";
    std::string endDateStr = "2025-01-01";
    std::string method = "sliding_window";
    int windowSize = 3;
    std::string dataPath = "/Users/hz9/dev/HAI_simulation/data/movements_cleaned_filled.csvv";
    std::string numEntryPath = "/Users/hz9/dev/HAI_simulation/movement_generation/deid_data/entries/num_entries.csv";
    std::string durationPath = "/Users/hz9/dev/HAI_simulation/movement_generation/deid_data/durations/durations.csv";
    int numSample = 1;

    runGeneration(
        numSample,
        transitionMatrixFolderPath,
        outputFolderPath,
        startDateStr,
        endDateStr,
        method,
        windowSize,
        dataPath,
        false,
        10,
        true,
        numEntryPath,
        durationPath
    );

    return 0;
}