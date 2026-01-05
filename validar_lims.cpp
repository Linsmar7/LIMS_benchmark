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
#include <cmath>
#include <unistd.h>

using namespace std;
namespace fs = std::filesystem;

// --- Ground Truth Structures ---
struct PointEntry {
  unsigned int id;
  vector<double> coords;
  double dist;
};

// --- Helper Functions ---

double getRAMUsageMB() {
    long virtual_size = 0L;
    long rss = 0L;
    ifstream stat_stream("/proc/self/statm", ios_base::in);
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

double getL2(const vector<double>& p1, const vector<double>& p2) {
  double sum = 0;
  for (size_t i = 0; i < p1.size(); ++i) {
    double diff = p1[i] - p2[i];
    sum += diff * diff;
  }
  return sqrt(sum);
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

int main(int argc, const char* argv[]){
    srand((int)time(0));
    if(argc < 5){
        cout << "Usage: ./validar_lims [dataset_name] [number of reference point] [number of cluster] [dimention of point] [index_dir (optional)]" << endl;
        return 0;
    }

    string dataset_arg = argv[1];
    fs::path argPath(dataset_arg);
    string dataset_name = argPath.stem().string();
    
    unsigned num_ref = stoi(argv[2]);
    unsigned num_clu = stoi(argv[3]);
    unsigned dim = stoi(argv[4]);
    string index_dir = "./inputFiles";
    if(argc > 5) index_dir = argv[5];

    // --- 1. Load Ground Truth (Full Dataset) ---
    cout << "Loading Ground Truth Dataset..." << endl;
    string datasetPath = dataset_arg;

    if (!fs::exists(datasetPath)) {
        vector<string> searchPaths = {
            "./datasets/" + dataset_arg + ".txt",
            "../datasets/" + dataset_arg + ".txt",
            "data/" + dataset_arg + ".txt",
            "datasets_processed/Imagenet32_train/" + dataset_arg + ".txt"
        };
        
        bool found = false;
        for (const auto& path : searchPaths) {
            if (fs::exists(path)) {
                datasetPath = path;
                found = true;
                break;
            }
        }
        
        if (!found) {
             cout << "Dataset file not found: " << datasetPath << " (and checked common search paths)" << endl;
             return 1;
        }
    }

    vector<PointEntry> fullData;
    ifstream infile(datasetPath);
    string line; 
    unsigned int currentId = 0;
    
    while (getline(infile, line)) {
        stringstream ss(line); string val; vector<double> c;
        while (getline(ss, val, ',')) c.push_back(stod(val));
        if (c.size() == dim) fullData.push_back({currentId++, c, 0.0});
    }
    cout << "Loaded " << fullData.size() << " points for Ground Truth." << endl;


    // --- 2. Load LIMS Index ---
    cout << "Loading LIMS Index..." << endl;
    vector<Clu_Point> all_data;
    vector<mainRef_Point> all_refSet;

    vector<Point> pivots;
    pivots.reserve(num_clu);
    vector<vector<Point> > oth_pivots;
    oth_pivots.reserve(num_clu);
    
    // Load Pivots (Ref.txt)
    string filename = index_dir + "/ref/ref.txt";
    ifstream fin;
    fin.open(filename);
    if(!fin){
        filename = index_dir + "/ref.txt";
        fin.open(filename);
        if(!fin) {
             cout << "Reference file not found." << endl;
             return 1;
        }
    }
    while(getline(fin,line)){
        stringstream ss(line); 
        string value;
        vector<double> coordinate;
        while(getline(ss, value, ',')) coordinate.push_back(atof(value.c_str()));
        pivots.push_back(Point(coordinate));
    }
    fin.close();

    // Load Other Pivots
    for(unsigned p = 0; p < num_clu; ++p){
        string filename = index_dir + "/ref/ref_" + to_string(p) +".txt";
        ifstream fin;
        fin.open(filename);
        if(!fin){
             filename = index_dir + "/ref_" + to_string(p) + ".txt";
             fin.open(filename);
             if(!fin) exit(1);
        }
        string line;
        vector<Point> other_pivot;
        while(getline(fin,line)){
            stringstream ss(line); 
            string value;
            vector<double> coordinate;
            while(getline(ss, value, ',')) coordinate.push_back(atof(value.c_str()));
            other_pivot.push_back(Point(coordinate));
        }
        oth_pivots.push_back(other_pivot);
        fin.close();
    }

    // Build/Load Clusters Metadata
    for(unsigned i = 0; i < num_clu; i++){
        string clu_file = index_dir + "/clu/8d_" + to_string(i) + ".txt";
        if (!fs::exists(clu_file)) clu_file = index_dir + "/clu_" + to_string(i) + ".txt";
        if (!fs::exists(clu_file) && index_dir == "./inputFiles") clu_file = "./inputFiles/clu/8d_" + to_string(i) + ".txt";
        
        InputReader inputReader(clu_file);
        all_data.push_back(inputReader.getCluster());

        if(all_data[i].clu_point.empty()) return 0;

        ChooseRef ref_point(num_ref - 1, all_data[i], pivots[i],oth_pivots[i],1);
        CaculateIValue calIValue(all_data[i], ref_point.getMainRefPoint());

        all_data[i] = calIValue.getCluster();
        all_refSet.push_back(calIValue.getMainRef_Point());
    }

    // --- 3. Run Validation Queries ---
    string queryPath = "r_tree/queries/" + dataset_name + "_knn.csv";
    if (!fs::exists(queryPath)) {
        cout << "Query file not found: " << queryPath << endl;
        return 0;
    }
    vector<Point> list_KNN = LoadPointForQuery(queryPath);
    
    // Output file
    if (!fs::exists("results")) fs::create_directory("results");
    string resultsFile = "results/validacao_lims_" + dataset_name + ".csv";
    ofstream log(resultsFile);
    log << "Query_ID,Tipo,K,Tempo_ms,Paginas_Lidas,Recall,Resultados_Encontrados\n";

    unsigned K = 5;
    
    // Allocate pt_status
    typedef int8_t Array20k[20000];
    Array20k* pt_status = new Array20k[num_clu];

    for(unsigned m = 0; m < list_KNN.size(); ++m){
        // --- A. EXACT SEARCH (Ground Truth) ---
        // Calc distances
        for (auto& entry : fullData) {
            entry.dist = getL2(entry.coords, list_KNN[m].coordinate);
        }
        // Sort
        // Optimization: partial_sort or nth_element for top K
        partial_sort(fullData.begin(), fullData.begin() + K, fullData.end(), [](const PointEntry& a, const PointEntry& b) {
            return a.dist < b.dist;
        });

        vector<unsigned int> groundTruthIds;
        for (unsigned int j = 0; j < K; ++j) groundTruthIds.push_back(fullData[j].id);


        // --- B. LIMS SEARCH ---
        int page = 0;
        double init_r = 0.05;
        double delta_r = 0.05;

        // Reset pt_status
        for(unsigned i=0; i<num_clu; ++i) 
             for(unsigned j=0; j<20000; ++j) pt_status[i][j] = 0;

        chrono::steady_clock::time_point q_begin = chrono::steady_clock::now();
        priority_queue<pair<double, unsigned int> > KNNRes_queue;
        
        while(KNNRes_queue.size() < K || KNNRes_queue.top().first > init_r + delta_r){
            for(unsigned i = 0; i < num_clu; ++i){
                CalCirclePos mainRefPtCircle(all_refSet[i].point, all_refSet[i].r, list_KNN[m], init_r);
                if(mainRefPtCircle.label == 1) continue;
                if(mainRefPtCircle.label == 3 && mainRefPtCircle.dis_upper < all_refSet[i].r_low) continue;
                if(mainRefPtCircle.label == 3 && init_r > all_refSet[i].r){
                    for(unsigned j = 0; j < all_refSet[i].iValuePts.size(); ++j){
                        double dis_pt_qrpt = CaculateEuclideanDis(all_refSet[i].iValuePts[j], list_KNN[m]);
                        if(KNNRes_queue.size() < K){
                            pt_status[i][j] = 1;
                            KNNRes_queue.push({dis_pt_qrpt, all_refSet[i].iValuePts[j].id});
                        }else{
                            if(KNNRes_queue.top().first > dis_pt_qrpt && pt_status[i][j] == 0){
                                pt_status[i][j] = 1;
                                KNNRes_queue.pop();
                                KNNRes_queue.push({dis_pt_qrpt, all_refSet[i].iValuePts[j].id});
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
                    if(RefPtCircle.label == 1){ flag = false; break; }
                    if(RefPtCircle.label == 3 && RefPtCircle.dis_upper < all_refSet[i].ref_points[j].r_low){ flag = false; break; }
                    if(RefPtCircle.label == 3 && init_r > all_refSet[i].ref_points[j].r){
                        for(unsigned l = 0; l < all_refSet[i].iValuePts.size(); ++l){
                            double dis_pt_qrpt = CaculateEuclideanDis(all_refSet[i].iValuePts[l], list_KNN[m]);
                            if(KNNRes_queue.size() < K){
                                pt_status[i][l] = 1;
                                KNNRes_queue.push({dis_pt_qrpt, all_refSet[i].iValuePts[l].id});
                            }else{
                                if(KNNRes_queue.top().first > dis_pt_qrpt && pt_status[i][l] == 0){
                                    pt_status[i][l] = 1;
                                    KNNRes_queue.pop();
                                    KNNRes_queue.push({dis_pt_qrpt, all_refSet[i].iValuePts[l].id});
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
            init_r += delta_r;
        }
        
        // Final precise check (copy from benchmark)
        for(unsigned i = 0; i < num_clu; ++i){
             CalCirclePos mainRefPtCircle(all_refSet[i].point, all_refSet[i].r, list_KNN[m], init_r);
             if(mainRefPtCircle.label == 1) continue;
             if(mainRefPtCircle.label == 3 && mainRefPtCircle.dis_upper < all_refSet[i].r_low) continue;
             if(mainRefPtCircle.label == 3 && init_r > all_refSet[i].r){
                  for(unsigned j = 0; j < all_refSet[i].iValuePts.size(); ++j){
                       double dis_pt_qrpt = CaculateEuclideanDis(all_refSet[i].iValuePts[j], list_KNN[m]);
                       if(KNNRes_queue.size() < K){
                            pt_status[i][j] = 1;
                            KNNRes_queue.push({dis_pt_qrpt, all_refSet[i].iValuePts[j].id});
                       }else{
                            if(KNNRes_queue.top().first > dis_pt_qrpt && pt_status[i][j] == 0){
                                 KNNRes_queue.pop();
                                 KNNRes_queue.push({dis_pt_qrpt, all_refSet[i].iValuePts[j].id});
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
                  if(RefPtCircle.label == 1){ flag = false; break; }
                  if(RefPtCircle.label == 3 && RefPtCircle.dis_upper < all_refSet[i].ref_points[j].r_low){ flag = false; break; }
                  if(RefPtCircle.label == 3 && init_r > all_refSet[i].ref_points[j].r){
                       for(unsigned l = 0; l < all_refSet[i].iValuePts.size(); ++l){
                            double dis_pt_qrpt = CaculateEuclideanDis(all_refSet[i].iValuePts[l], list_KNN[m]);
                            if(KNNRes_queue.size() < K){
                                 pt_status[i][l] = 1;
                                 KNNRes_queue.push({dis_pt_qrpt, all_refSet[i].iValuePts[l].id});
                            }else{
                                 if(KNNRes_queue.top().first > dis_pt_qrpt && pt_status[i][l] == 0){
                                      pt_status[i][l] = 1;
                                      KNNRes_queue.pop();
                                      KNNRes_queue.push({dis_pt_qrpt, all_refSet[i].iValuePts[l].id});
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


        // --- C. RECALL CALCULATION ---
        // Extract IDs from queue (it's a max heap, so top is largest distance)
        vector<unsigned int> limsIds;
        // Copy queue to not destroy it (or just pop)
        priority_queue<pair<double, unsigned int> > tempQ = KNNRes_queue;
        while(!tempQ.empty()){
            limsIds.push_back(tempQ.top().second);
            tempQ.pop();
        }

        int matches = 0;
        for(unsigned int id : limsIds){
             for(unsigned int gtId : groundTruthIds){
                  if(id == gtId) {
                       matches++;
                       break;
                  }
             }
        }
        double recall = (double)matches / K;
        
        log << m << ",kNN," << K << "," << q_time << "," << page << "," << recall << "," << limsIds.size() << "\n";
        cout << "Query " << m << ": Recall=" << recall << " Time=" << q_time << "ms" << endl;
    }

    // --- D. RANGE QUERY VALIDATION ---
    string rangeQueryPath = "r_tree/queries/" + dataset_name + "_range.csv";
    if (!fs::exists(rangeQueryPath)) {
        cout << "Range query file not found: " << rangeQueryPath << endl;
    } else {
        vector<Point> list_rangeQry = LoadPointForQuery(rangeQueryPath);
        double range_r = 0.1; // Default radius, MUST match benchmark settings
        
        cout << "\nRunning " << list_rangeQry.size() << " Range queries (r=" << range_r << ")..." << endl;
        
        for(unsigned m = 0; m < list_rangeQry.size(); ++m){
             // --- 1. GROUND TRUTH (Linear Scan) ---
             vector<unsigned int> gtIds;
             for (auto& entry : fullData) {
                  double d = getL2(entry.coords, list_rangeQry[m].coordinate);
                  if (d <= range_r) {
                       gtIds.push_back(entry.id);
                  }
             }

             // --- 2. LIMS RANGE SEARCH ---
             vector<int> rangeQueryRes;
             rangeQueryRes.reserve(10000);
             double io_time = 0.0;
             int page_range = 0;

             chrono::steady_clock::time_point start_r = chrono::steady_clock::now();

             for(unsigned i = 0; i < num_clu; ++i){
                CalCirclePos mainRefPtCircle(all_refSet[i].point, all_refSet[i].r, list_rangeQry[m], range_r);
                if(mainRefPtCircle.label == 1) continue;
                if(mainRefPtCircle.label == 3 && mainRefPtCircle.dis_upper < all_refSet[i].r_low) continue;
                if(mainRefPtCircle.label == 3 && range_r > all_refSet[i].r){
                    // Direct scan of cluster
                    ifstream fin;
                    int start_page = 0;
                    int end_page = all_refSet[i].iValuePts.size() / Constants::PAGE_SIZE;
                    for(int s = start_page; s <= end_page; ++s){
                        string fileName = "./data/cluster_" + to_string(i) + "_" + to_string(s);
                        fin.open(fileName, ios::binary);
                        if (!fin) continue; 
                        ++page_range;
                        int size_page;
                        fin.read((char*)&size_page,4);
                        int size_dim;
                        fin.read((char*)&size_dim,4);
                        
                        vector<vector<double> > pt_page;
                        pt_page.reserve(size_page);
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
                    CalCirclePos RefPtCircle(all_refSet[i].ref_points[j].point, all_refSet[i].ref_points[j].r, list_rangeQry[m], range_r);
                    if(RefPtCircle.label == 1){ flag = false; break; }
                    ref_query.push_back(RefPtCircle);
                }

                if(!flag) continue;

                RangeQuery_IO rangeQuery( list_rangeQry[m], range_r, all_refSet[i], rangeQueryRes, mainRefPtCircle, ref_query,i,page_range,io_time);
             }
             
             chrono::steady_clock::time_point end_r = chrono::steady_clock::now();
             double time_r = chrono::duration_cast<chrono::microseconds>(end_r - start_r).count() / 1000.0;

             // --- 3. RECALL ---
             sort(rangeQueryRes.begin(), rangeQueryRes.end());
             sort(gtIds.begin(), gtIds.end());
             
             // Intersection
             vector<int> intersection;
             set_intersection(rangeQueryRes.begin(), rangeQueryRes.end(),
                              gtIds.begin(), gtIds.end(),
                              back_inserter(intersection));
             
             double recall = 0.0;
             if (gtIds.size() > 0) recall = (double)intersection.size() / gtIds.size();
             else recall = 1.0;

             log << m << ",Range," << range_r << "," << time_r << "," << page_range << "," << recall << "," << rangeQueryRes.size() << "\n";
             cout << "Range Query " << m << ": Recall=" << recall << " Time=" << time_r << "ms (Found " << rangeQueryRes.size() << "/" << gtIds.size() << ")" << endl;
        }
    }

    delete[] pt_status;
    log.close();
    cout << "Validation saved to " << resultsFile << endl;
    return 0;
}
