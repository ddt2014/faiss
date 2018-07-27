#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <chrono>
#include <fstream>
#include <stdarg.h>  // For va_start, etc.
#include <memory>    // For std::unique_ptr
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <algorithm>
#include <fcntl.h>
#include <stack>
#include <ctime>
#include <sys/time.h>
#include <time.h>
#include <thread>
#include <boost/filesystem.hpp>
#include <json.hpp>
#include <faiss/c_api/IndexFlat.h>
#include <faiss/c_api/IndexIVFPQ.h>
#include "records.hpp"
#include "types.hpp"

using namespace std;
using namespace tunicornface;
using namespace std::chrono;
namespace bfs = boost::filesystem;

std::stack<high_resolution_clock::time_point> tictoc_stack;

void tic() {
  tictoc_stack.push(high_resolution_clock::now());
}

double toc() {
  cout << "Time elapsed: ";
  double diff = duration_cast<milliseconds>( high_resolution_clock::now() - tictoc_stack.top() ).count();
  cout << diff << " ms";
  tictoc_stack.pop();
  return diff;
}

bool read_txt(string listfilename, vector<string> &filenames, vector<int> &labels) {
  ifstream infile;
  cout<<listfilename<<endl;
  string line;
  infile.open(listfilename);
  if(!infile.is_open()) {
  cout << "can't open file" << listfilename << endl;
  return false;
  } else {
  cout<<"opened list file " << listfilename << endl;
  }

  while (getline(infile, line))
  {
  vector<string> strings;
  boost::trim(line);
  filenames.push_back(line);
  }
  infile.close();
  return true;
}

bool ReadRecordsFromList(string db_list, Record &records){
  vector<string> filenames;
  vector<int> labels;
  if(!read_txt(db_list, filenames, labels)) {// read rec file names
  cout << "read txt failed" << endl;
  return false;
  }

  for(unsigned int i = 0; i < filenames.size(); i++){
  cout << filenames[i] << endl;
  Record rec; // read each rec file in some way...
  ReadRecords(filenames[i], rec);  
  records.filenames.insert(records.filenames.end(), rec.filenames.begin(), rec.filenames.end());
  records.features.insert(records.features.end(), rec.features.begin(), rec.features.end());  
  }  
  return true;
}

template <typename T>
T* vec2ptr(const vector<vector<T> >& vecs) {
  unsigned long long cnt = 0;
  T* ptrs = new T[vecs.size() * vecs[0].size()];

  for(size_t i = 0; i < vecs.size(); ++i) {
      memcpy((ptrs + cnt), vecs[i].data(), vecs[i].size() * sizeof(T));
      cnt += vecs[i].size();
  }
  return ptrs;
}

int main() {
  double diff;
  struct timeval begin, end;
  json::JSON js;
  js["search_results"]=json::Array();

  string query_list = "/storage2/public/Data/Surveilliance/features/v1.5_0629/query_list0629.txt";
  string search_list = "/home/dingxu/faiss/build/bin/search_db_list0629.txt";
  string train_list = "/storage2/public/Data/Surveilliance/features/v1.5_0629/train_list0629.txt";
  string output_dir = "/home/dingxu/faiss/tutorial/cpp";

  Record records_query;
  ReadRecordsFromList(query_list, records_query);
  Record records_search;
  ReadRecordsFromList(search_list, records_search);
  Record records_train;
  ReadRecordsFromList(train_list, records_train);

  float *x_train = vec2ptr(records_train.features);
  float *x_query = vec2ptr(records_query.features);
  float *x_search = vec2ptr(records_search.features);

  int d = records_train.features[0].size();              // dimension
  int nt = records_train.features.size();             // database size
  int ns = records_search.features.size();             // database size
  int nq = records_query.features.size();            // nb of queries
  int k = 100;
  faiss::IndexFlatL2 index(d);     // the other index

  tic();
  index.add(ns, x_search);
  diff = toc();
  cout << " add time" << endl;

  {     // search xq
    long *I = new long[k * nq], cnt = 0, cnt2 = 0;
    double sum_time = 0;
    float *D = new float[k * nq];
/*    gettimeofday(&begin, NULL);
    index.search(nq, x_query, k, D, I);
    gettimeofday(&end, NULL);
    diff = (end.tv_sec - begin.tv_sec) * 1000.0 + (end.tv_usec - begin.tv_usec) / 1000.0;
    cout << " search time: " << diff << endl;
*/
    for(int i = 0; i < nq; i++) {
      gettimeofday(&begin, NULL);
      index.search(1, x_query + cnt, k, D + cnt2, I + cnt2);
      gettimeofday(&end, NULL);
      diff = (end.tv_sec - begin.tv_sec) * 1000.0 + (end.tv_usec - begin.tv_usec) / 1000.0;
      sum_time += diff;
      cout << " search time, " << sum_time / (i + 1) << " ms, number: " << i << endl;
      json::JSON query;
      query["query_filename"]=records_query.filenames[i];
      json::JSON pq_topn = json::Array();
      for(int j = 0; j < k; j++){
        json::JSON tmp;
        tmp["filename"] = records_search.filenames[I[i * k + j]];
        tmp["score"] = D[i * k + j];
        pq_topn.append(tmp);
      }
      query["pq_search_result"] = pq_topn;
      query["pq_search_time"] = diff;
      js["search_results"].append(query);
      cnt += d;
      cnt2 += k;
    }
    
    string ouput_json_filename = output_dir + "/result.json";
    ofstream out(ouput_json_filename);
    out << js;
    out.close();

    delete [] I;
    delete [] D;
  }

  delete [] x_train;
  delete [] x_query;
  delete [] x_search;
  return 0;
}
