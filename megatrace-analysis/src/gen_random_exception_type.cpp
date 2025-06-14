#include<iostream>
#include<fstream>
using namespace std;

const int rank_num = 8;
const int min_log_count = 260;
const int max_log_count = 2100;


void writeToFile(const string& filename, int rank_id, string type, int log_count) {
    ofstream outFile;
    
    outFile.open(filename, ios::app);

    if (!outFile) {
        cerr << "[LOGGER] cannot open file: " << filename << endl;
        return;
    }
    outFile << rank_id << "," << type << "," << log_count << endl;

    outFile.close();
}

int main(int argc, char* argv[]){
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <random_param_data_size> <param_file_path>" << endl;
        return 1;
    }

    int exception_random_data_size = stoi(argv[1]);
    string output_file_path = argv[2];

    ofstream clearFile(output_file_path, ios::trunc);
    if (!clearFile) {
        cerr << "cannot open file: " << output_file_path << endl;
        return 1;
    }
    clearFile.close();

    srand(static_cast<unsigned int>(time(nullptr)));

    while(exception_random_data_size--){
        int exception_type_idx = rand()%2;
        int exception_log_count = rand()%(max_log_count - min_log_count + 1) + min_log_count;
        int exception_rank_id = rand()%8;
        string exception_type = exception_type_idx == 0 ? "slow" : "hang";
        writeToFile(output_file_path, exception_rank_id, exception_type, exception_log_count);
    }
    cout << "[LOGGER] random param loaded : " << output_file_path << endl;

    return 0;
}