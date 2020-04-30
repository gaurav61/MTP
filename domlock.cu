#include <bits/stdc++.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;
using namespace std::chrono;

/*  Global Variables    */

int *edge_array,*edge_array_parent,*vertex_array,*vertex_array_parent,*start_interval,*end_interval;
bool *active,*explored,*parent_updated,*is_leaf;
int counter=0;




/*  GPU Methods begins here  */

//__device__ volatile int *mutex;
//cudaMalloc((void **)&mutex,sizeof(int));
//cudaMemset(mutex, 0, sizeof(int));


__device__ float generate(curandState* globalState, int ind)
{
    //int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init ( seed, id, 0, &state[id] );
}

__device__ int findDominator(int *request_array,int actual_num_of_requests,int *logical_interval_x,int *logical_interval_y,int *vertex_array_d,int number_of_nodes,int *edge_array_d,int edges,int root){
    int ma=INT_MIN;
    int mi=INT_MAX;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i=0;i<actual_num_of_requests;i++){
        if(logical_interval_y[request_array[i]]>ma){
            ma=logical_interval_y[request_array[i]];
        }
        if(logical_interval_x[request_array[i]]<mi){
            mi=logical_interval_x[request_array[i]];
        }
    }
    //printf("%d  %d  %d\n",id,mi,ma);
    int neighbours,ptr;
    while(true){
        ptr=root;
        neighbours=vertex_array_d[ptr+1]-vertex_array_d[ptr];
        for(int i=0;i<neighbours;i++){
            if((logical_interval_x[edge_array_d[vertex_array_d[ptr]+i]]<=mi) && (logical_interval_y[edge_array_d[vertex_array_d[ptr]+i]]>=ma)){
                ptr=edge_array_d[vertex_array_d[ptr]+i];
                //printf("%d\n",id);
                break;
            }
        }
        if((logical_interval_x[root]==logical_interval_x[ptr]) && (logical_interval_y[root]==logical_interval_y[ptr])){
            break;
        }
        else{
            root=ptr;
        }
    //    printf("hmmmmm\n");
    }
    return root;

}


// *mutex should be 0 before calling this function
__global__ void Domlock(volatile int *thread_pool_x,volatile int *thread_pool_y,int *logical_interval_x,int *logical_interval_y,int num_of_threads,int *edge_array_d,int edges,int *vertex_array_d,int number_of_nodes,int root,volatile int *mutex,curandState* globalState) 
{
   	int id = threadIdx.x + blockIdx.x * blockDim.x; //Calculating thread id
	int batch_size=1;
    /*    Additional features of version 2 begins here     */

	int max_num_of_requests=5;
        int request_array[5];
        int temp;
	
	bool isSet;
	bool overlap1;
	bool overlap2;

	for(int bs=0;bs<batch_size;bs++){
	
        isSet = false;     //To check the critical section
	overlap1 = true;   //To check for overlap after read 1
    	overlap2 = false;  //To check for overlap after read 2

        int actual_num_of_requests=max_num_of_requests;
        for(int i=0;i<actual_num_of_requests;i++){
            temp=generate(globalState, id)*10000000;
            request_array[i]=temp%number_of_nodes;
	    //printf("%d : %d  %d  %d\n",id,request_array[i],logical_interval_x[request_array[i]],logical_interval_y[request_array[i]]);
        }

        int request_node=findDominator(request_array,actual_num_of_requests,logical_interval_x,logical_interval_y,vertex_array_d,number_of_nodes,edge_array_d,edges,root);
	//request_node=0;
	//printf("%d : %d\n",id,request_node);
    /*   Additional features of version 2 ends here        */

    int x=logical_interval_x[request_node]; //The interval which needs to be locked
    int y=logical_interval_y[request_node];
    //printf("%d\n",request_node);
    //printf("%d  %d\n",x,y);
    do 
    {
	//printf("%d\n",id);
        //Traverse the pool here and check for overlap
        overlap1=true;
        for(int i=0;i<num_of_threads;i++){ //Read 1
            int a=thread_pool_x[i];
            int b=thread_pool_y[i];
            if((a<=x && b>=x)||(a<=y && b>=y)||(x<a && y>b)){   //Overlap exists
                overlap1=false;
                //printf("%d\n",id);
		//printf("A %d %d %d\n",id,a,b);
		//printf("X %d %d %d\n",id,x,y);
                break;
            }
        }

        if (overlap1 && (isSet = atomicCAS((int *)mutex, 0, 1) == 0)) 
        {
            // critical section goes here
           // printf("%d\n",id);
            int flag=0;
            for(int i=0;i<num_of_threads;i++){ //Read 2
                int a=thread_pool_x[i];
                int b=thread_pool_y[i];
                if((a<=x && b>=x)||(a<=y && b>=y)||(x<a && y>b)){   //Overlap exists 
                    flag=1;
                    //printf("%d\n",id);
                    break;
                }
            }
            if(!flag){  //No overlap after read 2
                overlap2=true;
		//printf("OK %d\n",id);
                thread_pool_x[id]=x;    //Making the entry in the thread pool
                thread_pool_y[id]=y;
            }
            
        }
        if (isSet)  //if acquired the lock then release it
        {
	      //printf("%d\n",id);
             *mutex = 0;
        }
        if(overlap2){   //if this particular thread was able to lock a particular node
            for(int i=0;i<10000;i++){
		//printf("hmm\n");

                //do nothing just to waste some cycles
		//printf("Wasting some cycles\n");
            }
            thread_pool_x[id]=0;    //release the node
            thread_pool_y[id]=0;    
        }
	
    } 
    while (!overlap2);  //if the thread was successfull in locking the required node then exit the loop

  }
    
}
/*  GPU Methods ends here  */


//#################################################################################################

/*  CPU Methods begins here  */

void update_parent(int parent,int node){
	if((start_interval[parent]==0)&&(end_interval[parent]==0)){ //if the node is not updated before
		start_interval[parent]=start_interval[node];
		end_interval[parent]=end_interval[node];
	}
	else if((start_interval[parent]==start_interval[node])&&(end_interval[parent]==end_interval[node])){    //if the new update is same as previous
		return;
	}
	else{   //update the node
		if(start_interval[parent]>start_interval[node]){
			start_interval[parent]=start_interval[node];
		}
		if(end_interval[parent]<end_interval[node]){
			end_interval[parent]=end_interval[node];
		}
	}

	if(parent_updated[parent]){ //if this node has updated his parent then recursively update the parent
		int root_neighbours_parent=vertex_array_parent[parent+1]-vertex_array_parent[parent];
		int root_index_parent=vertex_array_parent[parent];
		for(int i=root_index_parent;i<root_index_parent+root_neighbours_parent;i++){
			update_parent(edge_array_parent[i],parent);	
			}
	}
}

void DFS(int root){
	//visited[root]=true;
	if(!explored[root]){
		if(is_leaf[root]||active[root]){
			counter++;
			start_interval[root]=counter;
			end_interval[root]=counter;
		}
		else{
			active[root]=true;
			int root_neighbours=vertex_array[root+1]-vertex_array[root];
			int root_index=vertex_array[root];
			for(int i=root_index;i<root_index+root_neighbours;i++){
				DFS(edge_array[i]);
			}
		}
		explored[root]=true;
		active[root]=false;
	}

	int root_neighbours_parent=vertex_array_parent[root+1]-vertex_array_parent[root];
	int root_index_parent=vertex_array_parent[root];
	for(int i=root_index_parent;i<root_index_parent+root_neighbours_parent;i++){
			update_parent(edge_array_parent[i],root);		
			}
	parent_updated[root]=true;
}

void DFShelper(int root,int nodes){

	printf("DFS begins here...\n");
	

	DFS(root);

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
	edge_array=new int[edges];
	vertex_array=new int[nodes+1];
	edge_array_parent=new int[edges];
	vertex_array_parent=new int[nodes+1];
	start_interval=new int[nodes];
	end_interval=new int[nodes];
	active=new bool[nodes];
	explored=new bool[nodes];
	parent_updated=new bool[nodes];
	is_leaf=new bool[nodes];


	for(int i=0;i<nodes;i++){
		parent_updated[i]=false;
		explored[i]=false;
		active[i]=false;
		is_leaf[i]=false;
		start_interval[i]=0;
		end_interval[i]=0;
	}


}

/*  CPU Methods ends here  */

int main(){

    int nodes,edges,root;
    int num_of_threads,num_of_blocks;
	cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
	unordered_map<int,vector<int> >m,m2;
	cin>>nodes>>edges;
	int u,v;
	for(int i=0;i<edges;i++){
		cin>>u>>v;
		m[u].push_back(v);
		m2[v].push_back(u);
    }
    cin>>root;
    cin>>num_of_threads>>num_of_blocks;
    cout<<nodes<<"  "<<edges<<endl;
    init(nodes,edges);
    CSR(m,vertex_array,edge_array,nodes);
    CSR(m2,vertex_array_parent,edge_array_parent,nodes);
    find_leaf(m,nodes);
    auto begin_time = high_resolution_clock::now();
    DFShelper(root,nodes);
    auto end_time = high_resolution_clock::now();
    auto dfs_time = duration_cast<microseconds>(end_time-begin_time);
    cout<<"DFS time taken : "<<dfs_time.count()<<endl;

    /*for(int ga=0;ga<nodes;ga++){
	printf("%d  %d  %d\n",ga,start_interval[ga],end_interval[ga]);
    }*/


    int *thread_pool_x_h,*thread_pool_x_d,*thread_pool_y_h,*thread_pool_y_d,*start_interval_d,*end_interval_d;
    int *edge_array_d,*vertex_array_d;
    int *mutex_h,*mutex_d;

    thread_pool_x_h=new int[num_of_threads*num_of_blocks]();
    thread_pool_y_h=new int[num_of_threads*num_of_blocks]();

    for(int i=0;i<num_of_threads*num_of_blocks;i++){
	thread_pool_x_h[i]=0;
	thread_pool_y_h[i]=0;
    }

    mutex_h=new int[1];
    mutex_h[0]=0;


    cudaMalloc(&thread_pool_x_d,num_of_threads*num_of_blocks*sizeof(int));
    cudaMemcpy(thread_pool_x_d,thread_pool_x_h,num_of_threads*num_of_blocks*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&thread_pool_y_d,num_of_threads*num_of_blocks*sizeof(int));
    cudaMemcpy(thread_pool_y_d,thread_pool_y_h,num_of_threads*num_of_blocks*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&start_interval_d,nodes*sizeof(int));
    cudaMemcpy(start_interval_d,start_interval,nodes*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&end_interval_d,nodes*sizeof(int));
    cudaMemcpy(end_interval_d,end_interval,nodes*sizeof(int),cudaMemcpyHostToDevice);


    /*    Additional features of version 2 begins here     */

    cudaMalloc(&edge_array_d,edges*sizeof(int));
    cudaMemcpy(edge_array_d,edge_array,edges*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&vertex_array_d,(nodes+1)*sizeof(int));
    cudaMemcpy(vertex_array_d,vertex_array,(nodes+1)*sizeof(int),cudaMemcpyHostToDevice);


    /*    Additional features of version 2 ends here     */

    cudaMalloc(&mutex_d,sizeof(int));
    cudaMemcpy(mutex_d,mutex_h,sizeof(int),cudaMemcpyHostToDevice);

    curandState* devStates;
    cudaMalloc (&devStates, num_of_threads * sizeof(curandState));
    srand(time(0));

    int seed = rand();
    setup_kernel<<<num_of_blocks, num_of_threads>>>(devStates,seed);
    int num_of_trials=10;
    float gpu_time_used=0.0;
    for(int i=0;i<num_of_trials;i++){
    cudaEventRecord(start);
    Domlock<<<num_of_blocks,num_of_threads>>>(thread_pool_x_d,thread_pool_y_d,start_interval_d,end_interval_d,num_of_threads*num_of_blocks,edge_array_d,edges,vertex_array_d,nodes,root,mutex_d,devStates);
    //cudaThreadSynchronize();
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds=0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    //cout<<milliseconds<<endl;
    gpu_time_used=gpu_time_used+milliseconds;
    }
    cout<<gpu_time_used/num_of_trials<<endl;
    printf("Completed...\n");

    return 0;

}
