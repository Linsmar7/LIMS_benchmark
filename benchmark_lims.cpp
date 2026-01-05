#include "fileIO/FileIO.h"
#include "caculateRef/ChooseRef.h"
#include "caculateRef/CaculateIValue.h"
#include "entities/Reference.h"
#include "entities/Point.h"
#include "queryProcessing/RangeQuery.h"
#include "queryProcessing/PointQuery.h"
#include "queryProcessing/RangeQuery_IO.h"
#include "queryProcessing/PointQuery_IO.h"
#include "queryProcessing/KNN.h"
#include "queryProcessing/KNN_IO.h"
#include "common/CalCirclePos.h"

#include "dis.h"
#include <sys/stat.h>
#include <sys/types.h>

#include <string>
#include <chrono>
#include <ctime>
#include <sstream>
#include <queue>
#include <algorithm>
#include <filesystem>

// #define random(a,b) (rand() % (b-a+1))+a
// #define rand() ((rand()%10000)*10000+rand()%10000)

using namespace std;
namespace fs = std::filesystem;
#include <unistd.h>

// Obtém o uso de RAM atual em MB
double getRAMUsageMB() {
    long virtual_size = 0L;
    long rss = 0L;
    ifstream stat_stream("/proc/self/statm", ios_base::in);
    // Lê o primeiro valor (VSize) e depois o segundo (RSS)
    if (stat_stream >> virtual_size >> rss) {
        return (rss * sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
    }
    return 0.0;
}

double CaculateEuclideanDis(Point &point_a, Point &point_b){
    double total = 0.0;
    for(unsigned i = 0; i < point_a.coordinate.size(); i++){
        total += pow(point_a.coordinate[i] - point_b.coordinate[i], 2);
    }
    return sqrt(total);
}

// Helper to get total size of a directory or set of files
uintmax_t getTreeSize(const string& index_dir) {
    uintmax_t totalSize = 0;
    
    // 1. Count reference files and cluster indices in index_dir
    if (fs::exists(index_dir)) {
        for (const auto& entry : fs::directory_iterator(index_dir)) {
            string filename = entry.path().filename().string();
            // Count ref.txt, ref_*.txt, clu_*.txt
            if (filename.find("ref") != string::npos || filename.find("clu") != string::npos) {
                 if (fs::is_regular_file(entry.status())) {
                     totalSize += fs::file_size(entry);
                 }
            }
        }
    }

    // 2. Count data pages in ./data
    string dataDir = "./data";
    if (fs::exists(dataDir)) {
        for (const auto& entry : fs::directory_iterator(dataDir)) {
            if (fs::is_regular_file(entry.status())) {
                totalSize += fs::file_size(entry);
            }
        }
    }
    
    return totalSize;
}

double CaculateEuclideanDis2(vector<double> &point_a, Point &point_b){
    double total = 0.0;
    for(unsigned i = 0; i < point_a.size(); i++){
        total += pow(point_a[i] - point_b.coordinate[i], 2);
    }
    return sqrt(total);
}

vector<Point> LoadPointForQuery(string filename){
    ifstream fin;
    vector<Point> queryPoints;
    fin.open(filename.c_str());
    if(!fin){
        cout << filename << " file could not be opened\n";
        exit(0);
    }
    string line;
    while(getline(fin,line)){
        stringstream ss(line); 
        string value;
        vector<double> coordinate;
        while(getline(ss, value, ','))
            coordinate.push_back(atof(value.c_str()));

        Point point = Point(coordinate);
        queryPoints.push_back(point);
    }
    fin.close();
    return queryPoints;
}

static bool AscendingSort(const InsertPt& point_a, const InsertPt& point_b){
    return point_a.i_value < point_b.i_value;
}

int main(int argc, const char* argv[]){
    srand((int)time(0));
    if(argc < 5){
        cout << "Usage: ./benchmark_lims [dataset_name] [number of reference point] [number of cluster] [dimention of point] [index_dir (optional)]" << endl;
        return 0;
    }

    string dataset_name = argv[1];
    unsigned num_ref = stoi(argv[2]);
    unsigned num_clu = stoi(argv[3]);
    unsigned dim = stoi(argv[4]);
    string index_dir = "./inputFiles";
    if(argc > 5) index_dir = argv[5];

    if(num_ref > dim + 1){
        cout << "The number of reference point should be smaller than dimentional of point data. Plz reinput parameter" << endl;
        return 0;
    }

    // all data
    vector<Clu_Point> all_data;
    vector<mainRef_Point> all_refSet;

    vector<Point> pivots;
    pivots.reserve(num_clu);
    vector<vector<Point> > oth_pivots;
    oth_pivots.reserve(num_clu);
    string filename = index_dir + "/ref/ref.txt";
    ifstream fin;
    fin.open(filename);
    if(!fin){
        filename = index_dir + "/ref.txt";
        fin.open(filename);
        if(!fin) {
             cout << index_dir << "/ref/ref.txt or " << index_dir << "/ref.txt could not be opened\n";
             exit(0);
        }
    }
    string line;
    while(getline(fin,line)){
        stringstream ss(line); 
        string value;
        vector<double> coordinate;
        while(getline(ss, value, ',')){
            coordinate.push_back(atof(value.c_str()));
        }
        Point pivot_pt = Point(coordinate);
        pivots.push_back(pivot_pt);
    }
    fin.close();

    for(unsigned p = 0; p < num_clu; ++p){
        string filename = index_dir + "/ref/ref_" + to_string(p) +".txt";
        ifstream fin;
        fin.open(filename);
        if(!fin){
             // Try flat
             filename = index_dir + "/ref_" + to_string(p) + ".txt";
             fin.open(filename);
             if(!fin){
                cout << filename << " file could not be opened\n";
                exit(0);
             }
        }
        string line;
        vector<Point> other_pivot;
        while(getline(fin,line)){
            stringstream ss(line); 
            string value;
            vector<double> coordinate;
            while(getline(ss, value, ',')){
                coordinate.push_back(atof(value.c_str()));
            }
            Point pivot_pt = Point(coordinate);
            other_pivot.push_back(pivot_pt);
        }
        oth_pivots.push_back(other_pivot);
        fin.close();
    }


    double build_time = 0.0;
    for(unsigned i = 0; i < num_clu; i++){
        // Use clu_ prefix which is standard output of LIMS build
        string clu_file = index_dir + "/clu/8d_" + to_string(i) + ".txt";
        if (!fs::exists(clu_file)) {
             clu_file = index_dir + "/clu_" + to_string(i) + ".txt"; // Flat structure from build output
        }
        // Fallback for original hardcoded path if default dir
        if (!fs::exists(clu_file) && index_dir == "./inputFiles") {
             clu_file = "./inputFiles/clu/8d_" + to_string(i) + ".txt";
        }
        
        InputReader inputReader(clu_file);
        all_data.push_back(inputReader.getCluster());

        if(all_data[i].clu_point.empty()){
            cout << "Plz do not load a null file: " << clu_file << endl;
            return 0;
        }

        if(all_data[i].clu_point[0].coordinate.size() != dim) {
            cout << "Error: Dataset dimension mismatch. Index file " << clu_file 
                 << " has dim=" << all_data[i].clu_point[0].coordinate.size() 
                 << " but CLI arg dim=" << dim << endl;
            return 0;
        }

        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        ChooseRef ref_point(num_ref - 1, all_data[i], pivots[i],oth_pivots[i],1);
        CaculateIValue calIValue(all_data[i], ref_point.getMainRefPoint());
        chrono::steady_clock::time_point end = chrono::steady_clock::now();

        build_time += chrono::duration_cast<chrono::milliseconds>(end - begin).count();

        all_data[i] = calIValue.getCluster();
        all_refSet.push_back(calIValue.getMainRef_Point());
    }

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    
    // Create data directory if it doesn't exist
    if (!fs::exists("data")) {
        fs::create_directory("data");
    }

    for(unsigned i = 0; i < num_clu; i++){
        OutputPrinter output(i, all_refSet[i]);
    }
    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    build_time += chrono::duration_cast<chrono::milliseconds>(end - begin).count();

    // cout << "build time is : " << build_time << endl;

    unsigned K = 5;
    int page = 0;

    // Load Queries from r_tree/queries
    string queryPath = "r_tree/queries/" + dataset_name + "_knn.csv";
    if (!fs::exists(queryPath)) {
        cout << "Query file not found: " << queryPath << endl;
        return 0;
    }
    vector<Point> list_KNN = LoadPointForQuery(queryPath);
    
    // Output file
    if (!fs::exists("results")) {
        fs::create_directory("results");
    }
    string resultsFile = "results/benchmark_lims_" + dataset_name + ".csv";
    ofstream log(resultsFile);
    log << "Query_ID,Tipo,K_ou_Raio,Tempo_ms,Paginas_Lidas,RAM_MB,Resultados_Encontrados\n";



    // Allocate pt_status once outside the loop
    typedef int8_t Array20k[20000];
    Array20k* pt_status = new Array20k[num_clu];

    for(unsigned m = 0; m < list_KNN.size(); ++m){
        unsigned long long total_pages = 0;

        double init_r = 0.05;
        double delta_r = 0.05;
        for(unsigned i=0; i<num_clu; ++i) 
             for(unsigned j=0; j<20000; ++j) pt_status[i][j] = 0;

        chrono::steady_clock::time_point q_begin = chrono::steady_clock::now();

        priority_queue<pair<double, unsigned int> > KNNRes_queue;
        
        while(KNNRes_queue.size() < K || KNNRes_queue.top().first > init_r + delta_r){

            for(unsigned i = 0; i < num_clu; ++i){
                CalCirclePos mainRefPtCircle(all_refSet[i].point, all_refSet[i].r, list_KNN[m], init_r);
                if(mainRefPtCircle.label == 1){
                    continue;
                }
                if(mainRefPtCircle.label == 3 && mainRefPtCircle.dis_upper < all_refSet[i].r_low){
                    continue;
                }
                if(mainRefPtCircle.label == 3 && init_r > all_refSet[i].r){
                    for(unsigned j = 0; j < all_refSet[i].iValuePts.size(); ++j){
                        double dis_pt_qrpt = CaculateEuclideanDis(all_refSet[i].iValuePts[j], list_KNN[m]);
                        if(KNNRes_queue.size() < K){
                            pt_status[i][j] = 1;
                            pair<double, unsigned int> elem(dis_pt_qrpt,all_refSet[i].iValuePts[j].id);
                            KNNRes_queue.push(elem);
                        }else{
                            if(KNNRes_queue.top().first > dis_pt_qrpt && pt_status[i][j] == 0){
                                pt_status[i][j] = 1;
                                KNNRes_queue.pop();
                                pair<double, unsigned int> elem(dis_pt_qrpt,all_refSet[i].iValuePts[j].id);
                                KNNRes_queue.push(elem);
                            }
                        }
                    }
                    continue;
                }


                bool flag = true;
                vector<CalCirclePos> ref_query;
                ref_query.reserve(num_ref - 1);
                for(unsigned j = 0; j < num_ref - 1; ++j){
                    CalCirclePos RefPtCircle(all_refSet[i].ref_points[j].point, all_refSet[i].ref_points[j].r, list_KNN[m], init_r);
                    if(RefPtCircle.label == 1){
                        flag = false;
                        break;
                    }
                    if(RefPtCircle.label == 3 && RefPtCircle.dis_upper < all_refSet[i].ref_points[j].r_low){
                        flag = false;
                        break;
                    }
                    if(RefPtCircle.label == 3 && init_r > all_refSet[i].ref_points[j].r){
                        for(unsigned l = 0; l < all_refSet[i].iValuePts.size(); ++l){
                            double dis_pt_qrpt = CaculateEuclideanDis(all_refSet[i].iValuePts[l], list_KNN[m]);
                            if(KNNRes_queue.size() < K){
                                pt_status[i][l] = 1;
                                pair<double, unsigned int> elem(dis_pt_qrpt,all_refSet[i].iValuePts[l].id);
                                KNNRes_queue.push(elem);
                            }else{
                                if(KNNRes_queue.top().first > dis_pt_qrpt && pt_status[i][l] == 0){
                                    pt_status[i][l] = 1;
                                    KNNRes_queue.pop();
                                    pair<double, unsigned int> elem(dis_pt_qrpt,all_refSet[i].iValuePts[l].id);
                                    KNNRes_queue.push(elem);
                                }
                            }
                        }
                        break;
                    }
                    ref_query.push_back(RefPtCircle);
                }

                if(!flag)
                    continue;
                
                KNN_IO KNNQuery(list_KNN[m], all_refSet[i], KNNRes_queue, mainRefPtCircle, ref_query, K, pt_status, i, page);
            }

            init_r += delta_r;
        }

        // Final precise range query
        for(unsigned i = 0; i < num_clu; ++i){
            CalCirclePos mainRefPtCircle(all_refSet[i].point, all_refSet[i].r, list_KNN[m], init_r);
            if(mainRefPtCircle.label == 1){
                continue;
            }
            if(mainRefPtCircle.label == 3 && mainRefPtCircle.dis_upper < all_refSet[i].r_low){
                continue;
            }
            if(mainRefPtCircle.label == 3 && init_r > all_refSet[i].r){
                for(unsigned j = 0; j < all_refSet[i].iValuePts.size(); ++j){
                    double dis_pt_qrpt = CaculateEuclideanDis(all_refSet[i].iValuePts[j], list_KNN[m]);
                    if(KNNRes_queue.size() < K){
                        pt_status[i][j] = 1;
                        pair<double, unsigned int> elem(dis_pt_qrpt,all_refSet[i].iValuePts[j].id);
                        KNNRes_queue.push(elem);
                    }else{
                        if(KNNRes_queue.top().first > dis_pt_qrpt && pt_status[i][j] == 0){
                            KNNRes_queue.pop();
                            pair<double, unsigned int> elem(dis_pt_qrpt,all_refSet[i].iValuePts[j].id);
                            KNNRes_queue.push(elem);
                        }
                    }
                }
                continue;
            }


            bool flag = true;
            vector<CalCirclePos> ref_query;
            ref_query.reserve(num_ref - 1);
            for(unsigned j = 0; j < num_ref - 1; ++j){
                CalCirclePos RefPtCircle(all_refSet[i].ref_points[j].point, all_refSet[i].ref_points[j].r, list_KNN[m], init_r);
                if(RefPtCircle.label == 1){
                    flag = false;
                    break;
                }
                if(RefPtCircle.label == 3 && RefPtCircle.dis_upper < all_refSet[i].ref_points[j].r_low){
                    flag = false;
                    break;
                }
                if(RefPtCircle.label == 3 && init_r > all_refSet[i].ref_points[j].r){
                    for(unsigned l = 0; l < all_refSet[i].iValuePts.size(); ++l){
                        double dis_pt_qrpt = CaculateEuclideanDis(all_refSet[i].iValuePts[l], list_KNN[m]);
                        if(KNNRes_queue.size() < K){
                            pt_status[i][l] = 1;
                            pair<double, unsigned int> elem(dis_pt_qrpt,all_refSet[i].iValuePts[l].id);
                            KNNRes_queue.push(elem);
                        }else{
                            if(KNNRes_queue.top().first > dis_pt_qrpt && pt_status[i][l] == 0){
                                pt_status[i][l] = 1;
                                KNNRes_queue.pop();
                                pair<double, unsigned int> elem(dis_pt_qrpt,all_refSet[i].iValuePts[l].id);
                                KNNRes_queue.push(elem);
                            }
                        }
                    }
                    break;
                }
                ref_query.push_back(RefPtCircle);
            }

            if(!flag) continue;

             KNN_IO KNNQuery(list_KNN[m], all_refSet[i], KNNRes_queue, mainRefPtCircle, ref_query, K, pt_status, i, page);
        }

        chrono::steady_clock::time_point q_end = chrono::steady_clock::now();
        double q_time = chrono::duration<double, milli>(q_end - q_begin).count();
        
        log << m << ",kNN," << K << "," << q_time << "," << page << "," << getRAMUsageMB() << "," << K << "\n";
        page = 0;
    }
    
    delete[] pt_status;

    log.close();
    
    // --- Range Queries ---
    string rangeQueryPath = "r_tree/queries/" + dataset_name + "_range.csv";
    if (!fs::exists(rangeQueryPath)) {
        cout << "Range query file not found: " << rangeQueryPath << endl;
        // Not a fatal error, just skip
    } else {
        vector<Point> list_rangeQry = LoadPointForQuery(rangeQueryPath);
        double r = 0.1; // Default radius, should match benchmark_rstar (0.1)
        
        cout << "Running " << list_rangeQry.size() << " Range queries..." << endl;
        ofstream log_append(resultsFile, ios::app); 
        
        page = 0;
        for(unsigned m = 0; m < list_rangeQry.size(); ++m){
            vector<int> rangeQueryRes;
            rangeQueryRes.reserve(10000);
            double io_time = 0.0;
            page = 0;

            chrono::steady_clock::time_point begin = chrono::steady_clock::now();

            for(unsigned i = 0; i < num_clu; ++i){
                CalCirclePos mainRefPtCircle(all_refSet[i].point, all_refSet[i].r, list_rangeQry[m], r);
                if(mainRefPtCircle.label == 1){
                    continue;
                }
                if(mainRefPtCircle.label == 3 && mainRefPtCircle.dis_upper < all_refSet[i].r_low){
                    continue;
                }
                if(mainRefPtCircle.label == 3 && r > all_refSet[i].r){

                    ifstream fin;
                    int start_page = 0;
                    int end_page = all_refSet[i].iValuePts.size() / Constants::PAGE_SIZE;

                    for(int s = start_page; s <= end_page; ++s){
                        vector<vector<double> > pt_page;
                        string fileName = "./data/cluster_" + to_string(i) + "_" + to_string(s);
                        fin.open(fileName, ios::binary);
                        if (!fin) continue; 
                        
                        ++page;
                        int size_page;
                        fin.read((char*)&size_page,4);
                        pt_page.reserve(size_page);
                        int size_dim;
                        fin.read((char*)&size_dim,4);
                        for(int q = 0 ; q < size_page; q++){
                            vector<double> coordinate;
                            coordinate.resize(size_dim);
                            fin.read(reinterpret_cast<char*>(coordinate.data()),size_dim * sizeof(coordinate.front()));
                            pt_page.push_back(coordinate);
                        } 
                        fin.close();
                        for(int l = 0+s*Constants::PAGE_SIZE; l < 0+s*Constants::PAGE_SIZE+size_page; ++l)
                            rangeQueryRes.push_back(all_refSet[i].iValuePts[l].id);
                    }
                    continue;
                }

                bool flag = true;
                vector<CalCirclePos> ref_query;
                ref_query.reserve(num_ref - 1);
                for(unsigned j = 0; j < num_ref - 1; ++j){
                    CalCirclePos RefPtCircle(all_refSet[i].ref_points[j].point, all_refSet[i].ref_points[j].r, list_rangeQry[m], r);
                    if(RefPtCircle.label == 1){
                        flag = false;
                        break;
                    }
                    ref_query.push_back(RefPtCircle);
                }

                if(!flag)
                    continue;

                RangeQuery_IO rangeQuery( list_rangeQry[m], r, all_refSet[i], rangeQueryRes, mainRefPtCircle, ref_query,i,page,io_time);
            }
            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            double time = chrono::duration_cast<chrono::microseconds>(end - begin).count() / 1000.0; // ms

            log_append << m << ",Range," << r << "," << time << "," << page << "," << getRAMUsageMB() << "," << rangeQueryRes.size() << "\n";
        }
        log_append.close();
    }
    
    cout << "Results saved to " << resultsFile << endl;

    // --- REPORT CONSTRUCTION STATS ---
    cout << "\n--- RESUMO DE CONSTRUCAO ---" << endl;
    
    // Offline Build Time
    double offlineBuildTime = 0.0;
    ifstream buildInfo("./data/lims_build_info.txt");
    if (buildInfo.is_open()) {
        buildInfo >> offlineBuildTime;
        buildInfo.close();
        cout << "Tempo de Construção (Offline): " << offlineBuildTime / 1000.0 << " s" << endl;
    } else {
        cout << "Tempo de Construção (Offline): N/A (arquivo ./data/lims_build_info.txt nao encontrado)" << endl;
    }

    // Online Setup Time
    cout << "Tempo de Setup (Online): " << build_time / 1000.0 << " s" << endl;

    // Tree Size
    double sizeMB = getTreeSize(index_dir) / (1024.0 * 1024.0);
    cout << "Tamanho da Árvore em Disco: " << sizeMB << " MB" << endl;

    return 0;
}
