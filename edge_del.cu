#include <bits/stdc++.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#define MAXEDGES 3000000
using namespace std;
using namespace std::chrono;
//sample comment2
int *edge_array,*edge_array_parent,*vertex_array,*vertex_array_parent,*start_interval,*end_interval;
bool *is_leaf;
int counter=0;

__global__ void BFS2(int* off,int* edge,int* off2, int* edge2,int* current,int* size,int N,int E,volatile int* c_arr,int* c_size,int* dist,volatile int* D_start_interval,volatile int* D_end_interval,volatile int* mutexs){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < *size){
        int node = current[id];
        int start = off[node];
        int end = off[node+1];
        
        while(start<end){
            int child = edge[start];
            bool isSet = false;
            do 
    		{
    			//printf("hmm\n");
        		if (isSet = atomicCAS((int *)(mutexs+child), 0, 1) == 0) 
        		{
				int start2 = off2[child];
				int end2 = off2[child+1];
				int ma = INT_MIN;
				int mi = INT_MAX;
				while(start2<end2){
					int child2 = edge2[start2];
					mi = min(mi,D_start_interval[child2]);
					ma = max(ma,D_end_interval[child2]);
					start2++;
				}
        			bool flag=false;
        				if(D_start_interval[child]>mi){
        					D_start_interval[child]=mi;
        					flag=true;
        				}
        				if(D_end_interval[child]<ma){
        					D_end_interval[child]=ma;
        					flag=true;
        				}

        			if (flag){
                	int index = atomicAdd(c_size,1);
                	c_arr[index]= child;
            		}

        		}
        		if (isSet)  //if acquired the lock then release it
        		{
            		mutexs[child] = 0;
        		}
	
    		}while (!isSet);

            start++;  
        }
    }
}

__global__ void BFS(int* off,int* edge,int* current,int* size,int N,int E,volatile int* c_arr,int* c_size,int* dist,volatile int* D_start_interval,volatile int* D_end_interval,volatile int* mutexs){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < *size){
        int node = current[id];
        int start = off[node];
        int end = off[node+1];
        
        while(start<end){
            int child = edge[start];
            bool isSet = false;
            do 
    		{
    			//printf("hmm\n");
        		if (isSet = atomicCAS((int *)(mutexs+child), 0, 1) == 0) 
        		{
        			bool flag=false;
        			if(D_start_interval[child]==0 && D_end_interval[child]==0){
        				D_start_interval[child]=D_start_interval[node];
        				D_end_interval[child]=D_end_interval[node];
        			}
        			else{
        				if(D_start_interval[child]>D_start_interval[node]){
        					D_start_interval[child]=D_start_interval[node];
        					flag=true;
        				}
        				if(D_end_interval[child]<D_end_interval[node]){
        					D_end_interval[child]=D_end_interval[node];
        					flag=true;
        				}
        			}

        			if ( dist[child] < 0 || flag){
                	dist[child] = 0;
                	int index = atomicAdd(c_size,1);
                	c_arr[index]= child;
            		}

        		}
        		if (isSet)  //if acquired the lock then release it
        		{
            		mutexs[child] = 0;
        		}
	
    		}while (!isSet);

            start++;  
        }
    }
}


void CSR(unordered_map<int,vector<int> > &m,int *vertex_array,int *edge_array, int nodes){
	
	int curr_index=0;
	for(int i=0;i<nodes;i++){
		int num_of_edges=m[i].size();
		vertex_array[i]=curr_index;
		for(int j=0;j<num_of_edges;j++){
			edge_array[curr_index+j]=m[i][j];
		}
		curr_index+=num_of_edges;
	}
	vertex_array[nodes]=curr_index;
}

void find_leaf(unordered_map<int,vector<int> >&m,int nodes){
	for(int i=0;i<nodes;i++){
		if(m[i].size()==0){
			is_leaf[i]=true;
		}
	}
}

void init(int nodes,int edges){
	edge_array=new int[MAXEDGES];
	vertex_array=new int[nodes+1];
	edge_array_parent=new int[MAXEDGES];
	vertex_array_parent=new int[nodes+1];
	start_interval=new int[nodes];
	end_interval=new int[nodes];
	is_leaf=new bool[nodes];


	for(int i=0;i<nodes;i++){
		is_leaf[i]=false;
		start_interval[i]=0;
		end_interval[i]=0;
	}
}

int main(){


	int nodes,edges,root;
	srand(time(NULL));
	unordered_map<int,vector<int> >m,m2;
	cin>>nodes>>edges;
	int u,v;
	for(int i=0;i<edges;i++){
		cin>>u>>v;
		m[u].push_back(v);
		m2[v].push_back(u);
    }
    cin>>root;
    init(nodes,edges);
    CSR(m,vertex_array,edge_array,nodes);
    CSR(m2,vertex_array_parent,edge_array_parent,nodes);
    find_leaf(m,nodes);
    

    int* H_current_node = (int*)malloc(sizeof(int)*MAXEDGES);
    int counter=0;
    for(int i=0;i<nodes;i++){
    	if(is_leaf[i]){
    		H_current_node[counter]=i;
    		counter++;
    		start_interval[i]=counter;
    		end_interval[i]=counter;
    		//cout<<i<<endl;
    	}
    }
    /*for(int i=0;i<counter;i++){
    	printf("%d ",H_current_node[i]);
    }*/
    
    int* H_c_size = (int*)malloc(sizeof(int));
    *H_c_size = counter;
    int* H_visited = (int*)malloc(sizeof(int)*nodes);
    memset(H_visited,-1,sizeof(int)*nodes);

    int* H_mutexs = (int*)malloc(sizeof(int)*nodes);
    memset(H_mutexs,0,sizeof(int)*nodes);


    for(int i=0;i<nodes;i++){
    	if(is_leaf[i]){
    		H_visited[i]=0;
    	}
    	//printf("%d\n",H_mutexs[i]);
    }
    /*for(int i=0;i<nodes;i++){
        printf("%d, %d\n",i,H_visited[i]);
    }
    printf("sdfbdsjfbsjd\n");*/
    int* a0 = (int*)malloc(sizeof(int));
    *a0=0;

    int* a1 = (int*)malloc(sizeof(int));
    *a1=counter;

    int* D_offset;
    int* D_edges;
    int* D_visited;
    int* D_offset2;
    int* D_edges2;

    int* D_current_node1;
    int* D_c_size1;
    int* D_current_node2;

    int* D_mutexs;
    int* D_start_interval;
    int* D_end_interval;


    int* D_c_size2;

    cudaMalloc(&D_offset,sizeof(int)*(nodes+1));
    cudaMalloc(&D_offset2,sizeof(int)*(nodes+1));
    cudaMalloc(&D_visited,sizeof(int)*nodes);
    cudaMalloc(&D_edges,sizeof(int)*MAXEDGES);
    cudaMalloc(&D_edges2,sizeof(int)*MAXEDGES);
    cudaMalloc(&D_current_node1,sizeof(int)*MAXEDGES);
    cudaMalloc(&D_c_size1,sizeof(int));
    cudaMalloc(&D_current_node2,sizeof(int)*MAXEDGES);
    cudaMalloc(&D_c_size2,sizeof(int));

    cudaMalloc(&D_mutexs,sizeof(int)*nodes);
    cudaMalloc(&D_start_interval,sizeof(int)*nodes);
    cudaMalloc(&D_end_interval,sizeof(int)*nodes);

    cudaMemcpy(D_offset,vertex_array_parent,sizeof(int)*(nodes+1),cudaMemcpyHostToDevice);
    cudaMemcpy(D_offset2,vertex_array,sizeof(int)*(nodes+1),cudaMemcpyHostToDevice);
    cudaMemcpy(D_edges,edge_array_parent,sizeof(int)*MAXEDGES,cudaMemcpyHostToDevice);
    cudaMemcpy(D_edges2,edge_array,sizeof(int)*MAXEDGES,cudaMemcpyHostToDevice);
    cudaMemcpy(D_current_node1,H_current_node,sizeof(int)*MAXEDGES,cudaMemcpyHostToDevice);
    cudaMemcpy(D_visited,H_visited,sizeof(int)*nodes,cudaMemcpyHostToDevice);
    cudaMemcpy(D_c_size1,a1,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(D_c_size2,a0,sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(D_mutexs,H_mutexs,sizeof(int)*nodes,cudaMemcpyHostToDevice);
    cudaMemcpy(D_start_interval,start_interval,sizeof(int)*nodes,cudaMemcpyHostToDevice);
    cudaMemcpy(D_end_interval,end_interval,sizeof(int)*nodes,cudaMemcpyHostToDevice);
    //printf("hmm\n");
    int i=1;
    while(*H_c_size>0){
    	
        int numThreads = 512;
        int numBlocks = (*H_c_size+numThreads-1)/numThreads;


        if(i%2==1){
            //use array 1
            //printf("hmm\n");
            BFS<<<numBlocks,numThreads>>>(D_offset,D_edges,D_current_node1,D_c_size1,nodes,edges,D_current_node2,D_c_size2,D_visited,D_start_interval,D_end_interval,D_mutexs);

            cudaMemcpy(H_c_size,D_c_size2, sizeof(int),cudaMemcpyDeviceToHost);
            // reset the index
            cudaMemcpy(D_c_size1,a0,sizeof(int),cudaMemcpyHostToDevice);
          
        }
        else{
            //use array 2
            BFS<<<numBlocks,numThreads>>>(D_offset,D_edges,D_current_node2,D_c_size2,nodes,edges,D_current_node1,D_c_size1,D_visited,D_start_interval,D_end_interval,D_mutexs);
            
            cudaMemcpy(H_c_size,D_c_size1, sizeof(int),cudaMemcpyDeviceToHost);
            //reset index
            cudaMemcpy(D_c_size2,a0,sizeof(int),cudaMemcpyHostToDevice);

        }
        i++;
        
    }
    cudaMemcpy(H_visited,D_visited, sizeof(int)*nodes,cudaMemcpyDeviceToHost);
    //printf("hmm\n"); 
    for(int j=nodes-1;j>=0;j--){
        //printf("%d %d %d %d\n",i,H_visited[i],start_interval[i],end_interval[i]);
        if(H_visited[j]==-1){
        	//printf("hmm\n");
        	H_current_node[0]=j;
        	*a1=1;
        	*H_c_size=1;
        	cudaMemcpy(D_current_node1,H_current_node,sizeof(int)*MAXEDGES,cudaMemcpyHostToDevice);	// OPTIMIZATION POSSIBLE HERE. INSTEAD OF COPYING ENTIRE ARRAY JUST COPY FIRST ELEMENT
        	cudaMemcpy(D_c_size1,a1,sizeof(int),cudaMemcpyHostToDevice);
    		cudaMemcpy(D_c_size2,a0,sizeof(int),cudaMemcpyHostToDevice);
    		counter++;
    		cudaMemcpy(&D_start_interval[j], &counter, sizeof(int), cudaMemcpyHostToDevice);
    		cudaMemcpy(&D_end_interval[j], &counter, sizeof(int), cudaMemcpyHostToDevice);
    		

        	i=1;
    		while(*H_c_size>0){
    			//printf("%d  %d\n",*H_c_size,j);
        		int numThreads = 512;
        		int numBlocks = (*H_c_size+numThreads-1)/numThreads;


        		if(i%2==1){
            		//use array 1
            		//printf("hmm\n");
            		BFS<<<numBlocks,numThreads>>>(D_offset,D_edges,D_current_node1,D_c_size1,nodes,edges,D_current_node2,D_c_size2,D_visited,D_start_interval,D_end_interval,D_mutexs);

            		cudaMemcpy(H_c_size,D_c_size2, sizeof(int),cudaMemcpyDeviceToHost);
            		// reset the index
            		cudaMemcpy(D_c_size1,a0,sizeof(int),cudaMemcpyHostToDevice);
          
        		}
        		else{
            		//use array 2
            		BFS<<<numBlocks,numThreads>>>(D_offset,D_edges,D_current_node2,D_c_size2,nodes,edges,D_current_node1,D_c_size1,D_visited,D_start_interval,D_end_interval,D_mutexs);
            
            		cudaMemcpy(H_c_size,D_c_size1, sizeof(int),cudaMemcpyDeviceToHost);
            		//reset index
            		cudaMemcpy(D_c_size2,a0,sizeof(int),cudaMemcpyHostToDevice);

        		}
        		i++;
        
    		}

    		cudaMemcpy(H_visited,D_visited,sizeof(int)*nodes,cudaMemcpyDeviceToHost);



        }
    }
    cudaMemcpy(start_interval,D_start_interval,sizeof(int)*nodes,cudaMemcpyDeviceToHost);
    cudaMemcpy(end_interval,D_end_interval,sizeof(int)*nodes,cudaMemcpyDeviceToHost);
    //printf("hmm\n");
    counter = 0;
    int batch_size=512;
    int *addition_x = new int[batch_size];
    int *addition_y = new int[batch_size];
    for(int i=0;i<batch_size;i++){
    	int st = rand()%nodes;
	int si = m[st].size();
	if(si==0){
		i--;
		continue;
	}
	int en = m[st][int(rand()%si)];
	addition_x[i]=st;
	addition_y[i]=en;
    }
    //printf("hmm\n");
    auto start_t2 = high_resolution_clock::now();
    for(int i=0;i<batch_size;i++){
	if(start_interval[addition_x[i]]==start_interval[addition_y[i]]){
		H_current_node[counter] = addition_x[i];
		counter++;
		int neigh = m[addition_x[i]].size();
		int mi = INT_MAX;
		for(int j=0;j<neigh;j++){
			if(m[addition_x[i]][j]!=addition_y[i]){
				mi = min(mi,start_interval[m[addition_x[i]][j]]);
			}
		}
		if(mi!=INT_MAX){
			start_interval[addition_x[i]]=mi;
		}
	}
	if(end_interval[addition_x[i]]==end_interval[addition_y[i]]){
		H_current_node[counter] = addition_x[i];
		counter++;
		int neigh = m[addition_x[i]].size();
		int ma = INT_MIN;
		for(int j=0;j<neigh;j++){
			if(m[addition_x[i]][j]!=addition_y[i]){
				ma = max(ma,end_interval[m[addition_x[i]][j]]);
			}
		}
		if(ma!=INT_MIN){
			end_interval[addition_x[i]]=ma;
		}
	}
    }
    auto stop_t2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop_t2 - start_t2);
    cout<<"Time taken : "<<duration2.count()<<endl;

    *H_c_size = counter;
    memset(H_visited,-1,sizeof(int)*nodes);

    memset(H_mutexs,0,sizeof(int)*nodes);


    for(int i=0;i<batch_size;i++){
    		H_visited[addition_x[i]]=0;
    	//printf("%d\n",H_mutexs[i]);
    }
    /*for(int i=0;i<nodes;i++){
        printf("%d, %d\n",i,H_visited[i]);
    }
    printf("sdfbdsjfbsjd\n");*/
    *a0=0;

    *a1=counter;


    cudaMemcpy(D_offset,vertex_array_parent,sizeof(int)*(nodes+1),cudaMemcpyHostToDevice);
    cudaMemcpy(D_edges,edge_array_parent,sizeof(int)*MAXEDGES,cudaMemcpyHostToDevice);
    cudaMemcpy(D_current_node1,H_current_node,sizeof(int)*MAXEDGES,cudaMemcpyHostToDevice);
    cudaMemcpy(D_visited,H_visited,sizeof(int)*nodes,cudaMemcpyHostToDevice);
    cudaMemcpy(D_c_size1,a1,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(D_c_size2,a0,sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(D_mutexs,H_mutexs,sizeof(int)*nodes,cudaMemcpyHostToDevice);
    cudaMemcpy(D_start_interval,start_interval,sizeof(int)*nodes,cudaMemcpyHostToDevice);
    cudaMemcpy(D_end_interval,end_interval,sizeof(int)*nodes,cudaMemcpyHostToDevice);

    i=1;
    //printf("hmm\n");
    auto start_t = high_resolution_clock::now();
    while(*H_c_size>0){
    	
        int numThreads = 512;
        int numBlocks = (*H_c_size+numThreads-1)/numThreads;


        if(i%2==1){
            //use array 1
            //printf("hmm\n");
            BFS2<<<numBlocks,numThreads>>>(D_offset,D_edges,D_offset2,D_edges2,D_current_node1,D_c_size1,nodes,edges,D_current_node2,D_c_size2,D_visited,D_start_interval,D_end_interval,D_mutexs);

            cudaMemcpy(H_c_size,D_c_size2, sizeof(int),cudaMemcpyDeviceToHost);
            // reset the index
            cudaMemcpy(D_c_size1,a0,sizeof(int),cudaMemcpyHostToDevice);
          
        }
        else{
            //use array 2
            BFS2<<<numBlocks,numThreads>>>(D_offset,D_edges,D_offset2,D_edges2,D_current_node2,D_c_size2,nodes,edges,D_current_node1,D_c_size1,D_visited,D_start_interval,D_end_interval,D_mutexs);
            
            cudaMemcpy(H_c_size,D_c_size1, sizeof(int),cudaMemcpyDeviceToHost);
            //reset index
            cudaMemcpy(D_c_size2,a0,sizeof(int),cudaMemcpyHostToDevice);

        }
        i++;
        
    }
    auto stop_t = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop_t - start_t);
    cout<<"Time taken : "<<duration.count()<<endl;
    
    printf("--------------------------------------------------\n\n");
    cudaMemcpy(start_interval,D_start_interval,sizeof(int)*nodes,cudaMemcpyDeviceToHost);
    cudaMemcpy(end_interval,D_end_interval,sizeof(int)*nodes,cudaMemcpyDeviceToHost);
    //for(int i=nodes-1;i>=0;i--){
      //  printf("%d %d %d %d\n",i,H_visited[i],start_interval[i],end_interval[i]);
      //  }

        CSR(m,vertex_array,edge_array,nodes);
    	CSR(m2,vertex_array_parent,edge_array_parent,nodes);
    	find_leaf(m,nodes);


    
	return 0;
}

