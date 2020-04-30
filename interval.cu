#include <bits/stdc++.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;
using namespace std::chrono;
int *edge_array,*edge_array_parent,*vertex_array,*vertex_array_parent,*start_interval,*end_interval;
bool *is_leaf;
int counter=0;


__global__ void BFS(int* off,int* edge,int* current,int* size,int N,int E,volatile int* c_arr,int* c_size,int* dist,volatile int* D_start_interval,volatile int* D_end_interval,volatile int* mutexs){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < *size){
        int node = current[id]; //get the current node
        int start = off[node]; // index of first neighbour
        int end = off[node+1]; // index of last neighbour
        
        while(start<end){	//traverse all the neighbours
            int child = edge[start];
            bool isSet = false;
            do 
    		{
    			//printf("hmm\n");
        		if (isSet = atomicCAS((int *)(mutexs+child), 0, 1) == 0)	//critical section begins here 
        		{
        			bool flag=false;
        			if(D_start_interval[child]==0 && D_end_interval[child]==0){ // if the parent is not updated before
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

        			if ( dist[child] < 0 || flag){ //if the parent interval is updated
                			dist[child] = 0;
                			int index = atomicAdd(c_size,1);
                			c_arr[index]= child;	//add it to the array for further propogation
            			}

        		}	//end of critical section
        		if (isSet)  //if acquired the lock then release it
        		{
            			mutexs[child] = 0;
        		}
	
    		}while (!isSet);

            start++;  //next neighbour
        }
    }
}

void CSR(unordered_map<int,vector<int> > &m,int *vertex_array,int *edge_array, int nodes){	//generates CSR representation of the graph
	
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

void find_leaf(unordered_map<int,vector<int> >&m,int nodes){	//find the leaf nodes
	for(int i=0;i<nodes;i++){
		if(m[i].size()==0){
			is_leaf[i]=true;
		}
	}
}

void init(int nodes,int edges){		//initilization of arrays
	edge_array=new int[edges];
	vertex_array=new int[nodes+1];
	edge_array_parent=new int[edges];
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
	unordered_map<int,vector<int> >m,m2;
	cin>>nodes>>edges;	//input
	int u,v;
	for(int i=0;i<edges;i++){	//input
		cin>>u>>v;
		m[u].push_back(v);
		m2[v].push_back(u);
    }
    cin>>root;	//input
    init(nodes,edges);
    CSR(m,vertex_array,edge_array,nodes);
    CSR(m2,vertex_array_parent,edge_array_parent,nodes);
    find_leaf(m,nodes);
    int* H_current_node = (int*)malloc(sizeof(int)*edges);
    int counter=0;
    for(int i=0;i<nodes;i++){
    	if(is_leaf[i]){
    		H_current_node[counter]=i;	//store all the leaf nodes in this array to start the parallel BFS on GPU
    		counter++;	// it is use to track the size of the above array
    		start_interval[i]=counter;	// initialize the intervals for leaf nodes
    		end_interval[i]=counter;
    		//cout<<i<<endl;
    	}
    }
    /*for(int i=0;i<counter;i++){
    	printf("%d ",H_current_node[i]);
    }*/
    
    int* H_c_size = (int*)malloc(sizeof(int));
    *H_c_size = counter;	// the number of nodes in the H_current_node array
    int* H_visited = (int*)malloc(sizeof(int)*nodes);
    memset(H_visited,-1,sizeof(int)*nodes);	// visited array to keep track of whether some node is visited or not 

    int* H_mutexs = (int*)malloc(sizeof(int)*nodes);
    memset(H_mutexs,0,sizeof(int)*nodes);	// mutex array to implement critical section.


    for(int i=0;i<nodes;i++){
    	if(is_leaf[i]){
    		H_visited[i]=0;	//visit the leaf nodes
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

    int* D_current_node1;
    int* D_c_size1;
    int* D_current_node2;

    int* D_mutexs;
    int* D_start_interval;
    int* D_end_interval;


    int* D_c_size2;

    cudaMalloc(&D_offset,sizeof(int)*(nodes+1));
    cudaMalloc(&D_visited,sizeof(int)*nodes);
    cudaMalloc(&D_edges,sizeof(int)*edges);
    cudaMalloc(&D_current_node1,sizeof(int)*edges);
    cudaMalloc(&D_c_size1,sizeof(int));
    cudaMalloc(&D_current_node2,sizeof(int)*edges);
    cudaMalloc(&D_c_size2,sizeof(int));

    cudaMalloc(&D_mutexs,sizeof(int)*nodes);
    cudaMalloc(&D_start_interval,sizeof(int)*nodes);
    cudaMalloc(&D_end_interval,sizeof(int)*nodes);

    cudaMemcpy(D_offset,vertex_array_parent,sizeof(int)*(nodes+1),cudaMemcpyHostToDevice);
    cudaMemcpy(D_edges,edge_array_parent,sizeof(int)*edges,cudaMemcpyHostToDevice);
    cudaMemcpy(D_current_node1,H_current_node,sizeof(int)*edges,cudaMemcpyHostToDevice);
    cudaMemcpy(D_visited,H_visited,sizeof(int)*nodes,cudaMemcpyHostToDevice);
    cudaMemcpy(D_c_size1,a1,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(D_c_size2,a0,sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(D_mutexs,H_mutexs,sizeof(int)*nodes,cudaMemcpyHostToDevice);
    cudaMemcpy(D_start_interval,start_interval,sizeof(int)*nodes,cudaMemcpyHostToDevice);
    cudaMemcpy(D_end_interval,end_interval,sizeof(int)*nodes,cudaMemcpyHostToDevice);

    int i=1;
    auto start = high_resolution_clock::now();
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
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);
    cout<<"BFS time : "<<duration.count()<<endl;
    cudaMemcpy(H_visited,D_visited, sizeof(int)*nodes,cudaMemcpyDeviceToHost);
    
    for(int j=nodes-1;j>=0;j--){
        //printf("%d %d %d %d\n",i,H_visited[i],start_interval[i],end_interval[i]);
        if(H_visited[j]==-1){	//for the remaining unexplored nodes
        	//printf("hmm\n");
        	H_current_node[0]=j;
        	*a1=1;
        	*H_c_size=1;
        	cudaMemcpy(D_current_node1,H_current_node,sizeof(int)*edges,cudaMemcpyHostToDevice);
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
    for(int i=nodes-1;i>=0;i--){
        //printf("%d %d %d %d\n",i,H_visited[i],start_interval[i],end_interval[i]);
        }
	printf("%d %d\n",start_interval[0],end_interval[0]);


	return 0;
}
