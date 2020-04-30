#include<bits/stdc++.h>
using namespace std;
int main(int argc, char** argv){ 
	string file(argv[1]);
        ifstream input(file);
        string line;
        int counter=0;
	unordered_map<int,vector<int> >m;
	stack<int>st;
        while(getline(input,line)){
		int l = line.length();
		for(int i=0;i<l;i++){
			if(line[i]=='<' && line[i+1]!='/'){	//start of the tag
				if(!st.empty()){
					m[st.top()].push_back(counter);
					st.push(counter);
					counter++;
				}
				else{
					st.push(counter);
					counter++;
				}
			}
			else if(line[i]=='<' && line[i+1]=='/'){	//end of the tag
				if(!st.empty()){
					st.pop();
				}
			}
		}			

	}
	int nodes,edges=0;
	nodes=counter;
	for(auto i:m){
		edges+=m[i.first].size();
	}
	cout<<nodes<<" "<<edges<<endl;
	for(auto i:m){
		for(int j=0;j<m[i.first].size();j++){
			printf("%d %d\n",i.first,m[i.first][j]);
		}
	}
	cout<<0<<endl;
	return 0;
}
