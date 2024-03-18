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
    PathSampler(const std::string& transitionMatrixFolderPath) {
        // Read transition matrix files and populate the transitionMatrices map
        // Assuming each file is named "transition_matrix_<duration>.csv"
        for (int duration = 1; duration <= 100; ++duration) {
            std::string filePath = transitionMatrixFolderPath + "/transition_matrix_" + std::to_string(duration) + ".csv";
            std::ifstream file(filePath);
            if (file.is_open()) {
                std::string line;
                std::getline(file, line); // Skip the header line

                std::unordered_map<std::string, std::unordered_map<std::string, double> > matrix;
                while (std::getline(file, line)) {
                    std::stringstream ss(line);
                    std::string field;

                    std::getline(ss, field, ','); // From Department
                    std::string fromDept = field;

                    std::unordered_map<std::string, double> row;
                    while (std::getline(ss, field, ',')) {
                        std::string toDept = field;
                        std::getline(ss, field, ',');
                        double probability = std::stod(field);
                        row[toDept] = probability;
                    }

                    matrix[fromDept] = row;
                }

                transitionMatrices[duration] = matrix;
                file.close();
            }
        }
    }

    std::vector<std::string> sample(int duration, const std::string& method, int windowSize = 0) {
        std::vector<std::string> path;
        path.push_back("ADMISSION");

        std::string currentDept = "ADMISSION";
        for (int i = 1; i < duration; ++i) {
            std::unordered_map<std::string, std::unordered_map<std::string, double> > matrix;
            if (method == "sliding_window") {
                int start = std::max(1, i - windowSize + 1);
                int end = i + 1;
                matrix = getAverageMatrix(start, end);
            } else if (method == "shorter_only") {
                matrix = transitionMatrices[i];
            } else if (method == "longer_only") {
                matrix = transitionMatrices[duration];
            }

            std::string nextDept = sampleNextDepartment(currentDept, matrix);
            path.push_back(nextDept);
            currentDept = nextDept;
        }

        return path;
    }

private:
    std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<std::string, double> > > transitionMatrices;

    std::string sampleNextDepartment(const std::string& currentDept, const std::unordered_map<std::string, std::unordered_map<std::string, double> >& matrix) {
        std::vector<std::string> departments;
        std::vector<double> probabilities;
        for (const auto& entry : matrix.at(currentDept)) {
            departments.push_back(entry.first);
            probabilities.push_back(entry.second);
        }

        std::discrete_distribution<> distribution(probabilities.begin(), probabilities.end());
        std::random_device rd;
        std::mt19937 gen(rd());
        int index = distribution(gen);
        return departments[index];
    }

    std::unordered_map<std::string, std::unordered_map<std::string, double> > getAverageMatrix(int start, int end) {
        std::unordered_map<std::string, std::unordered_map<std::string, double> > averageMatrix;
        int count = end - start;

        for (int i = start; i < end; ++i) {
            const auto& matrix = transitionMatrices[i];
            for (const auto& entry : matrix) {
                const std::string& fromDept = entry.first;
                const auto& row = entry.second;
                for (const auto& cell : row) {
                    const std::string& toDept = cell.first;
                    double probability = cell.second;
                    averageMatrix[fromDept][toDept] += probability;
                }
            }
        }

        for (auto& entry : averageMatrix) {
            auto& row = entry.second;
            for (auto& cell : row) {
                cell.second /= count;
            }
        }

        return averageMatrix;
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
    std::string transitionMatrixFolderPath = "/Users/hz9/dev/HAI_simulation/movement_generation/deid_data/transition_matrices";
    std::string outputFolderPath = "/Users/hz9/dev/HAI_simulation/movement_generation/output";
    std::string startDateStr = "2024-01-01";
    std::string endDateStr = "2025-01-01";
    std::string method = "sliding_window";
    int windowSize = 3;
    std::string dataPath = "/Users/hz9/dev/HAI_simulation/data/movements_cleaned_filled.csvv";
    std::string numEntryPath = "/Users/hz9/dev/HAI_simulation/movement_generation/deid_data/entries/num_entries.csv";
    std::string durationPath = "/Users/hz9/dev/HAI_simulation/movement_generation/deid_data/durations/durations.csv";
    int numSample = 3;

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