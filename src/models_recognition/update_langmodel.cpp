#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <utility>
#include <fstream>
#include <regex>

using namespace std;

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
    if (!item.empty()) {
            elems.push_back(item);
        }
    }
    return elems;
}

template <class T>
void print(T s){ std::cout << s;};
template <class T>
void println(T s){ std::cout << s << std::endl;};
template <class T>
void println(vector<T> vec){
	for(auto& d : vec){
		print(d);
		print(",");
	}
	println("");
}


class WordPoint
{
public:
	WordPoint(std::string _word, float _point)
	:word(_word),point(_point){

	}
	std::string word;
	float point;
};

void updateDict(std::vector<WordPoint> updateList, std::string input_model_path, std::string output_model_path){
	//std::vector<WordPoint> allWordList;
	//allWordList.reserve(70000);

	// 読み込み
	// 書き換え
	std::ifstream input_dic(input_model_path);
	std::ofstream output_dic(output_model_path);
	std::string line;
	std::smatch match;
	std::regex get_word("(.*)\\+.*\\+.*");
	std::regex split_dict("(.*)	(.*	.*)");
	std::regex split_at_dict("(.*)	.*	(.*	.*	.*)");
	std::string word;

	int cnt=0;
	while(getline(input_dic, line)){
		if(cnt%1000==0)println(cnt);
		++cnt;
		bool flag_update = false;
		line = line.erase(line.size()-1); // つらい
		regex_match(line, match, get_word);
		word = match[1];

		//printf("a/%s/a\n", line.c_str());
		//std::cout<<" word="<<word<<std::endl;
		//if(cnt>10) break;

		auto res = std::find_if(updateList.begin(), updateList.end(),
			[&](WordPoint const& list){
				return list.word == word;
			}
		);

		if(res != updateList.end()){
			//if((*res).point > 0.0001){
			flag_update = true;
			if(line.find("@") == std::string::npos){
				regex_match(line, match, split_dict);
				output_dic << match[1] << "	@" << (*res).point << "	" << word << "	" << match[2] << std::endl;
			}else{
				std::cout << "oh!"<<std::endl;
				//regex_match(line, match, split_at_dict);
				output_dic << line << std::endl;
				//output_dic << match[1] << "	@" << (*res).point << "	" << match[2] << std::endl;
			}
			//}
		}else{
		}
		if(flag_update == false){
			output_dic << line << std::endl;
		}
	}
	input_dic.close();
	output_dic.close();
	//std::cout<<"res:"<<allWordList[10000].word<<std::endl;
}

int main (int argc, char* argv[]) {
	std::string distance_path = argv[1];
	std::string input_model_path = argv[2];
	std::string output_model_path = argv[3];
	std::string dict_path = argv[4];
	int mode = stoi(argv[5]);

	// 更新する単語と確率リスト生成
	std::vector<WordPoint> wordPoint_dict;

	println("Get distance");
	std::ifstream dict_stream(distance_path);
	std::string line;
	std::vector<std::string> buf;

	std::vector<std::string> word_list;
	std::vector<float> point_list;
	while(getline(dict_stream, line)){
		buf = split(line,',');
		word_list.emplace_back(buf[0]);
		point_list.emplace_back(std::stof(buf[1]));
	}

	size_t dic_size = word_list.size();
	float min_point = point_list[point_list.size()-11];
	float max_point = point_list[10];
	float cent_point = max_point - (max_point - min_point ) / 2.0;
	//std::cout<<cent_point<<","<<max_point<<","<<min_point<<std::endl;
	//return 0;

	if(mode==1){
		for(unsigned int i=0;i<dic_size;++i){
			std::string& word = word_list[i];
			float& point = point_list[i];
			//重み付け
			//point = (point - 0.20) < 0 ? 0.0 : (point - 0.20) * 20.0;
			//point = (point - 0.20) < 0 ? 0.0 : ((point-0.20) * 20) * ((point-0.20) * 20) * 2;
			//point = (point - 0.20) < 0 ? 0.0 : 2.0;
			point = (point - cent_point) * (1.0 / (max_point - cent_point)) * 3.5;
			if(point>3.5) point = 3.5;
			//if(point<0.0) point = 0.0;
			wordPoint_dict.emplace_back(WordPoint(word,point));
		}
	}else if(mode==2){
		for(int i=0;i<1200;++i){
			std::string& word = word_list[i];
			float& point = point_list[i];
			//重み付け
			point = (point - cent_point) * (1.0 / (max_point - cent_point)) * 2.0;
			if(point>2.0) point=2.0;
			point += 0.5;
			wordPoint_dict.emplace_back(WordPoint(word,point));
		}
		for(unsigned int i=1201;i<dic_size;++i){
			std::string& word = word_list[i];
			float& point = point_list[i];
			point = 0;
			wordPoint_dict.emplace_back(WordPoint(word,point));
		}
	}else if(mode==3){
		for(int i=0;i<1200;++i){
			std::string& word = word_list[i];
			float& point = point_list[i];
			//重み付け
			float under = point_list[1200];
			point = (point - under) * (1.0 / (max_point - under)) * 2.5;
			if(point>2.5) point = 2.5;
			point += 0.5;
			wordPoint_dict.emplace_back(WordPoint(word,point));
		}
		for(unsigned int i=1201;i<dic_size;++i){
			std::string& word = word_list[i];
			float& point = point_list[i];
			point = 0;
			wordPoint_dict.emplace_back(WordPoint(word,point));
		}
	}

	// pointを保存
	std::ofstream output_dic(dict_path + "/point_distribution.csv");
	for(auto const& point : point_list){
		output_dic << point << std::endl;
	}
	output_dic.close();

	// 更新
	println("Start update dict");
	updateDict(wordPoint_dict, input_model_path, output_model_path);
}

