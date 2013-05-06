#pragma once
#include <map>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cassert>
#include <valarray>
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>
#include <algorithm>
#include<string>
#include<utility>
using std::pair;
using std::size_t;
using std::slice;
using std::slice_array;
using std::gslice_array;
using std::valarray;
using std::string;
using std::map;
using std::make_pair;
using std::copy;
using std::vector;

class Matrix
{
public:

   typedef std::vector<pair<double, valarray<double>>> ValueVectorList;

public:
/**
 * @name Matrix - Empty constructor
 * @return Matrix
 */
  inline  Matrix();
/**
 * @name Matrix - Constructor with given height and width
 * @param height -  also the number of rows
 * @param width - also the number of cols
 * @return Matrix
 */
   inline Matrix(size_t height, size_t width);
/**
 * @name Matrix - Constructor with an initializer list, in which
 * elements are arranged by row-major order(same with C/C++
 * principal). If the number of list is less than the actual size of actual
 * matirx, zeros will be added at the end of matrix
 * @param width - specify width of the matrix represented by the
 * initializer list.
 * @param list -  list 
 * @return Matrix
 */
  
/**
 * @name Matrix - Copy constructor
 * @param other -  other 
 * @return Matrix
 */
   inline Matrix(const Matrix &other);
/**
 * @name Matrix - Move constructor
 * @param other - rvalue
 * @return Matrix
 */
   inline Matrix(Matrix &&other) ;
/**
 * @name Matrix - Construct a sub-matrix from other super-matrix.
 * @param other -  the super-matrix
 * @param row -  row
 * @param col -  col 
 * @return Matrix
 */
   inline Matrix(const Matrix &other, const slice &row, const slice &col);
/**
 * @name Matrix - Destructor
 * @return void
 */
   inline ~Matrix();
public://operator overload
/**
 * @name () - get the reference of a element in the matrix
 * @param row -  row 
 * @param col -  col 
 * @return double
 */
   inline double& operator()(size_t row, size_t col);
/**
 * @name () - get the copy of a element in the matrix
 * @param row -  row 
 * @param col -  col 
 * @return double
 */
   inline double operator()(size_t row, size_t col) const;
/**
 * @name () - get the reference of a sub-matrix from this matrix
 * @param row -  row 
 * @param col -  col 
 * @return gslice_array
 */
   inline gslice_array<double> operator()(const slice &row, const slice &col); 
/**
 * @name = - assignment
 * @param other -  other 
 * @return Matrix
 */
   inline Matrix& operator=(const Matrix &other);
/**
 * @name = - assignment
 * @param other -  other 
 * @return Matrix
 */
   inline Matrix& operator=(Matrix &&other);
/**
 * @name = - the all element of this matrix is assigned to the same value
 * @param value - the same value
 * @return Matrix
 */
   inline Matrix& operator=(double value);
/**
 * @name * - the real matrix production
 * @param other - the right matrix in this matrix production. A assert
 * will test whether the height of the right matrix is equal to the
 * width of this matrix
 */
   inline Matrix operator*(const Matrix &other) const;

   inline Matrix& operator /=(double value);

public://public basic operation
/**
 * @name row - get the reference of a row in this matrix
 * @param indexOfRow -  index Of Row 
 * @return slice_array
 */
   inline slice_array<double> row(size_t indexOfRow);
/**
 * @name row - get the copy of a row in this matrix
 * @param indexOfRow -  index Of Row 
 * @return valarray
 */
   inline valarray<double> row(size_t indexOfRow) const;
/**
 * @name col - get the reference of a column in this matrix
 * @param indexOfCol -  index Of Col 
 * @return slice_array
 */
   inline slice_array<double> col(size_t indexOfCol);
/**
 * @name col - get the copy of a column in this matrix
 * @param indexOfCol -  index Of Col 
 * @return valarray
 */
   inline valarray<double> col(size_t indexOfCol) const;
/**
 * @name GetWidth - get the width(also the number of columns) of this matrix
 * @return size_t
 */
   inline size_t GetWidth() const;
/**
 * @name GetHeight - get the height(also the number of rows) of this matrix
 * @return size_t
 */
   inline size_t GetHeight() const;
/**
 * @name IsSquared - test whether this matrix is square
 * @return bool
 */
   inline bool IsSquared() const;
/**
 * @name IsSymmetric - test whether this matrix is symmetric
 * @return bool
 */
   inline bool IsSymmetric() const;
/**
 * @name IsSameSizeWith - test whether a given matrix is same size
 * with this matrix
 * @param other -  other 
 * @return bool
 */
   inline bool IsSameSizeWith(const Matrix &other) const;
public://advanced functions
/**
 * @name FilterByValid - symmetric filtering
 * @param window -  window 
 * @return Matrix - filtered result
 */
   inline Matrix FilterByValid(const Matrix &window);
/**
 * @name GetEigenvalueList - Get all eigenvalue with ascending
 * order. The implementation is based on DSYEVR in lapack
 * @return valarray
 */
   inline valarray<double> GetEigenvalueList() ;

   inline ValueVectorList GetEigenValueAndVector() const;
/**
 * @name GetSpectralDensity - Get the spectral density of
 * eigenvalues in this matrix
 * @param slide - the width of slice in this drawing
 * @return string - the path of generated image
 */
   inline void GetSpectralDensity(vector<double> &x,vector<double> &y,double slide);
private://real data
   size_t width;
   size_t height;
   valarray<double> vdata;
};

double& Matrix::operator()(size_t row, size_t col)
{
   return vdata[col * height + row];
}

double Matrix::operator()(size_t row, size_t col) const
{
   return vdata[col * height + row];
}

size_t Matrix::GetWidth() const
{
   return width;
}

size_t Matrix::GetHeight() const
{
   return height;
}

Matrix::Matrix()
   :width(0), height(0), vdata(0)
{
}

Matrix::Matrix(size_t height, size_t width)
   : height(height), width(width), vdata(width * height)
{
}

Matrix::Matrix(Matrix &&other)
:width(other.width), height(other.height), vdata(other.vdata)
{   
}

Matrix::Matrix(const Matrix &other)
   :width(other.GetWidth()), height(other.GetHeight()), vdata(other.vdata)
{
}

Matrix::~Matrix()
{
}

Matrix Matrix::operator*(const Matrix &other) const
{
   assert(width == other.height);
   
   Matrix result(height, other.width);
   for(size_t row = 0; row < height; row++)
   {
      for(size_t col = 0; col < other.width; col++)
      {	 
	 result.vdata[col * height + row] =
	    (vdata[slice(row, width, height)] 
	     * other.vdata[slice(col * other.height, other.height, 1)]).sum();
      }
   }
   return result;
}

Matrix& Matrix::operator=(const Matrix &other)
{
   width = other.GetWidth();
   height = other.GetHeight();
   vdata = other.vdata;
   return *this;
}

Matrix& Matrix::operator=(Matrix &&other)
{
   width = other.GetWidth();
   height = other.GetHeight();
   vdata = std::move(other.vdata);
   return *this;
}

Matrix& Matrix::operator=(double value)
{
   vdata = value;
   return *this;
}

slice_array<double> Matrix::row(size_t indexOfRow)
{
   assert(indexOfRow < height);
   return vdata[slice(indexOfRow, width, height)];
}

valarray<double> Matrix::row(size_t indexOfRow) const
{
   assert(indexOfRow < height);
   return vdata[slice(indexOfRow, width, height)];
}

slice_array<double> Matrix::col(size_t indexOfCol)
{
   assert(indexOfCol < width);
   return vdata[slice(indexOfCol * height, height, 1)];
}

valarray<double> Matrix::col(size_t indexOfCol) const
{
   assert(indexOfCol < width);
   return vdata[slice(indexOfCol * height, height, 1)];
}

bool Matrix::IsSquared() const
{
   if(width == height)
      return true;
   else
      return false;
}

bool Matrix::IsSymmetric() const
{
   if(width != height)
      return false;
   for(size_t i = 0; i < width; i++)
   {
      if(abs(row(i) - col(i)).sum() != 0)
	 return false;
   }
   return true;
}

bool Matrix::IsSameSizeWith(const Matrix &other) const
{
   if(this->width == other.GetWidth() &&
      this->height == other.GetHeight())
      return true;
   else
      return false;
}


Matrix Matrix::FilterByValid(const Matrix &window)
{
   assert(window.GetWidth() <= GetWidth() &&
	  window.GetHeight() <= GetHeight());

   Matrix result(GetWidth() - window.GetWidth() + 1,
		 GetHeight() - window.GetHeight() + 1);
   
   for(size_t row = 0; row < result.GetHeight(); row++)
   {
      for(size_t col = 0; col < result.GetWidth(); col++)
      {
	 double sum = 0;
	 for(size_t row_w = 0; row_w < window.GetHeight(); row_w++)
	 {
	    for(size_t col_w = 0; col_w < window.GetWidth(); col_w++)
	    {
	       sum += (*this)(row + row_w, col + col_w);
	    }
	 }
	 result(row, col) = sum;
      }
   }
   return result;
}

//需要解决Fortran功能
valarray<double> Matrix::GetEigenvalueList()
{
   assert(IsSymmetric());
   vector<double> data(begin(vdata), end(vdata));
   //lapack parameter list
   char jobz = 'N';//compute eigenvalues only
   char range = 'A';//all eigenvalues will be found
   char uplo = 'U';//upper triangle of A is stored
   int n = GetWidth();
   int lda = GetWidth();
   double vl, vu;//no referenced when range = 'A' or 'I'
   int il, iu;//no referenced if range = 'A' or 'V'
   double abstol = 0;//absolute error tolerance for the eigenvalues
   int m;//the total number of eigenvalues found
   vector<double> w(n);//the first m elements contain the selected
                       //eigenvalues in ascending order
   vector<double> z(n*n);//not referenced when jobz = 'N', but this
			 //reserved space
   int ldz = n;//the leading dimension of the array z
   vector<int> isuppz(2 * n);//the support of the eigenvectors in Z
   int info;
   //got the optimal size of workspace
   int lwork = -1;//the dimension of the array work
   int liwork = -1;//the dimension of the array iwork
   vector<double> work(26 * n);//workspace
   vector<int> iwork(10 * n);//workspace
   //query the optimal size
   /*dsyevr_(&jobz, &range, &uplo, &n, data.data(), &lda, &vl, &vu, &il, &iu, &abstol,
	   &m, w.data(), z.data(), &ldz, isuppz.data(), work.data(), &lwork,
	   iwork.data(), &liwork, &info);*/
   if(info == 0)
   {//succeed
      lwork = (int)ceil(work[0]);
      liwork = iwork[0];
      //reallocate
      work.resize(lwork);
      iwork.resize(liwork);
   }
   else
   {
      return valarray<double>();
   }
   //compute
   /*dsyevr_(&jobz, &range, &uplo, &n, data.data(), &lda, &vl, &vu, &il, &iu, &abstol,
	   &m, w.data(), z.data(), &ldz, isuppz.data(), work.data(), &lwork,
	   iwork.data(), &liwork, &info);*/
   if(info == 0)
   {
      valarray<double> result(n);
      copy(w.begin(), w.end(), begin(result));
      return result;
   }
   else
      return valarray<double>();
}
//需要解决Fortran功能
Matrix::ValueVectorList Matrix::GetEigenValueAndVector() const
{
   assert(IsSymmetric());
   vector<double> data(begin(vdata), end(vdata));
   //lapack parameter list
   char jobz = 'V';//compute eigenvalues only
   char range = 'A';//all eigenvalues will be found
   char uplo = 'U';//upper triangle of A is stored
   int n = GetWidth();
   int lda = GetWidth();
   double vl, vu;//no referenced when range = 'A' or 'I'
   int il, iu;//no referenced if range = 'A' or 'V'
   double abstol = 0;//absolute error tolerance for the eigenvalues
   int m;//the total number of eigenvalues found
   vector<double> w(n);//the first m elements contain the selected
                       //eigenvalues in ascending order
   vector<double> z(n * n);//eigenvector in each culumn
   int ldz = n;//the leading dimension of the array z
   vector<int> isuppz(2 * n);//the support of the eigenvectors in Z
   int info;
   //got the optimal size of workspace
   int lwork = -1;//the dimension of the array work
   int liwork = -1;//the dimension of the array iwork
   vector<double> work(26 * n);//workspace
   vector<int> iwork(10 * n);//workspace
   //query the optimal size
  /* dsyevr_(&jobz, &range, &uplo, &n, data.data(), &lda, &vl, &vu, &il, &iu, &abstol,
	   &m, w.data(), z.data(), &ldz, isuppz.data(), work.data(), &lwork,
	   iwork.data(), &liwork, &info);*/
   if(info == 0)
   {//succeed
      lwork = (int)ceil(work[0]);
      liwork = iwork[0];
      //reallocate
      work.resize(lwork);
      iwork.resize(liwork);
   }
   else
   {
      return ValueVectorList();
   }
   //compute
   /*dsyevr_(&jobz, &range, &uplo, &n, data.data(), &lda, &vl, &vu, &il, &iu, &abstol,
	   &m, w.data(), z.data(), &ldz, isuppz.data(), work.data(), &lwork,
	   iwork.data(), &liwork, &info);*/
   if(info == 0)
   {
      ValueVectorList result;
      for(int i = 0; i < n; i++)
      {
	 valarray<double> temp(n);
	 copy(z.begin() + i * n, z.begin() + (i+1) * n, begin(temp));
	 result.push_back(make_pair(w[i], temp));
      }
      return result;
   }
   else
      return ValueVectorList();
}

void Matrix::GetSpectralDensity(vector<double> &x,vector<double> &y, double slide)
{
   map<double, double> distribution;
   auto lambdas = GetEigenvalueList();
   int start = (int)floor(lambdas[0]);
   int end = (int)ceil(lambdas[lambdas.size() - 1]);
   for(int i = 0; i < (end - start) / slide; i++)
   {
      distribution[double(start) + i * slide] = 0;
   }
   for(size_t i = 0; i < lambdas.size(); i++)
   {
      distribution[floor(lambdas[i] / slide) * slide]++;
   }
   //normalized
   for(auto iter = distribution.begin(); iter != distribution.end(); iter++)
   {
     x.push_back(iter->first);
	 y.push_back(iter->second / lambdas.size());
	
   }
}

Matrix& Matrix::operator/=(double value)
{
   vdata /= value;
   return *this;
}


namespace scn //short for Su' Complex Networks
{
   extern "C++" class Graph;
   extern "C++" class UGraph;
   extern "C++" class DGraph;
//************************************************************
//public structures or typename
//************************************************************
   //template<typename Type> using Map = std::unordered_map<Type, Type>;
   //template<typename Type> using pMap = std::shared_ptr<Map<Type>>;
   
   //template<typename Type> using pVector = std::shared_ptr<std::vector<Type>>;

   enum{NaF = size_t(-1)};

   typedef std::vector<size_t>     IndexList;
   typedef std::vector<std::shared_ptr<UGraph>> UGraphList;
   typedef std::shared_ptr<UGraphList>          pUGraphList; 

   typedef unsigned long long int               uint64;
//************************************************************
//Definition of Graphs
//************************************************************

   template<class NodeType>
   class _node_iterator : public std::unordered_map<size_t, NodeType>::iterator
   {
   public:
      typedef NodeType   value_type;
      typedef size_t     reference;//dereference for the index of node
      typedef NodeType*  pointer;
   public:
      _node_iterator(){;}
      
      _node_iterator(typename std::unordered_map<size_t, NodeType>::iterator &&iter)
	 :iterator_base(iter){;}
   
      inline reference operator*() const
      {return iterator_base->second.GetIndexOfNode();}
      
      inline pointer operator->() const
      {return std::addressof(iterator_base->second);}
   
      inline _node_iterator& operator++()
      {
	 iterator_base++;
	 return *this;
      }
      
      inline _node_iterator& operator++(int)
      {
	 iterator_base++;
	 return *this;
      }

      inline _node_iterator operator+(int count)
      {
	 _node_iterator temp = *this;
	 for(int i = 0; i < count; i++)
	    temp.iterator_base++;
	 return temp;
      }
      
      inline bool operator==(const _node_iterator& other) const
      {
	 return iterator_base == other.iterator_base;
      }

      inline bool operator!=(const _node_iterator& other) const
      {
	 return iterator_base != other.iterator_base;
      }

      inline _node_iterator& operator=(const _node_iterator& other)
      {
	 iterator_base = other.iterator_base;
	 return *this;
      }
   protected:
      typename std::unordered_map<size_t, NodeType>::iterator iterator_base;
   };
//////////////////////////////////////////////////////////////
//////////////////////////

   class _node
   {
   public:
      typedef std::unordered_set<size_t>::const_iterator  inout_iterator;
   public:
      class iterator : public std::unordered_set<size_t>::iterator
      {
      public:
	 typedef size_t   value_type;
	 typedef const size_t&  reference;
	 typedef const size_t*  pointer;
	 typedef std::unordered_set<size_t>::iterator _base_type;
      public:
	 iterator(){}
	 
	 iterator(_base_type &&in_degree_begin)
	    :current(in_degree_begin){}
	 
	 inline reference operator*() const
	 {return reference(*current);}
	 
	 inline pointer operator->() const
	 {return pointer(*current);}
	 
	 inline iterator& operator++()
	 {
	    current++;
	    return *this;
	 }
	 inline iterator operator+(int count)
	 {
	    iterator other = *this;
	    for(int i = 0; i < count; i++)
	    {
	       other.current++;
	    }
	    return other;
	 }
	 inline iterator& operator++(int)
	 {
	    current++;
	    return *this;
	 }
	 
	 inline bool operator==(const iterator& other) const
	 { return current == other.current;}
	 
	 inline bool operator!=(const iterator& other) const
	 {  return current != other.current;}
	 
	 inline iterator& operator=(const iterator& other)
	 {
	    current = other.current;
	    return *this;
	 }
      protected:
	 _base_type current;
      };     
   public:
      _node(size_t indexOfNode = -1) : index(indexOfNode){;}
      ~_node(){;}
   public:
      inline _node::iterator begin()
      {
		  /*
	 return _node::iterator(in_degree.begin(), in_degree.end(), 
				out_degree.begin());
				*/
		  return _node::iterator(in_degree.begin());
      }
  
      inline _node::inout_iterator in_begin()
      {return in_degree.begin(); }
      
      inline _node::inout_iterator out_begin()
      {return out_degree.begin(); }
      
      inline _node::iterator end()
      {
		  /*
		  return _node::iterator(out_degree.end(), in_degree.end(), out_degree.begin());
		  */
		  return _node::iterator(in_degree.end());
	   }
      
      inline _node::inout_iterator in_end()
      {return in_degree.end(); }
      
      inline _node::inout_iterator out_end()
      {return out_degree.end(); }
   public:
      inline size_t GetDegree(){return in_degree.size() + out_degree.size();}
      inline size_t GetInDegree(){return in_degree.size();}
      inline size_t GetOutDegree(){return out_degree.size();}
      inline size_t GetIndexOfNode(){return index;}
   protected:
      friend class Graph;
      friend class UGraph;
      friend class DGraph;
      size_t index;
      std::unordered_set<size_t> in_degree;/**<in_degree set form a
					    * complete endge sets
					    * with out_degree in DGraph */
      std::unordered_set<size_t> out_degree;/**<In UGraph, these
					     * out_degree has no meanings */
   };
//////////////////////////////////////////////////////////////
//////////////////////////

   class Graph
   {
   public://constant numbers
      enum {NaF = size_t(-1)};
      //static constexpr size_t NaF = -1;
      typedef _node_iterator<_node>   iterator;
      typedef std::shared_ptr<Graph>  pGraph;
   public://constructor and destructor
      
      Graph(){};

      Graph(size_t numberOfNodes)
      {
	 Create(numberOfNodes);
      };
      
      ~Graph(){};

   public://public operation
      
      inline void Create(size_t numberOfNodes)
      {
//	 reserve(numberOfNodes);
	 for(size_t i = 0; i < numberOfNodes; i++)
	 {
	    AddNode();
	 }
      };

      inline IndexList CopyIndexOfNodes()
      {
	 IndexList list;
	 list.reserve(GetNumberOfNodes());
	 for(auto iter = node_list.begin(); iter != node_list.end(); iter++)
	 {
	    list.push_back(iter->first);
	 }
	 return list;
      }

      inline bool HasNode(size_t index)
      {
	 if(node_list.find(index) == node_list.end())
	    return false;
	 else
	    return true;
      }

      inline Graph::iterator AddNode(size_t index)
      {
	 assert(!HasNode(index));
	 return iterator(node_list.insert(std::make_pair(index, _node(index))).first);
      }

      inline size_t AddNode()
      {
	 assert(!HasNode(node_list.size()));
	 size_t index = node_list.size();
	 node_list[index].index = index;
	 return index;
      }
/**
 * @name RemoveNode - Directly remove a node and all associated edges.
 * @param index -  index of node
 * @return std::pair<the list of complentary nodes of in edges, 
 * the list of complentary nodes of out edges>. If remove a node in
 * undirected graph, please use: pair<list, std::ignore> to receive
 * the result of this function
 */
      inline std::pair<IndexList, IndexList>
	 RemoveNode(size_t index)
      { 
	 assert(HasNode(index));
	 auto iter = node_list.find(index);
	 IndexList in, out;
	 
	 in.assign(iter->second.in_degree.begin(), iter->second.in_degree.end());
	 out.assign(iter->second.out_degree.begin(), iter->second.out_degree.end());
	
	 for(auto other = iter->second.begin();
	     other != iter->second.end(); other++)
	 {
	    node_list[*other].in_degree.erase(index);
	    node_list[*other].out_degree.erase(index);
	 }
	 
	 node_list.erase(iter);
	 return std::make_pair(std::move(in), std::move(out));
      }

      inline size_t GetNumberOfEdges()
      {
	 size_t sum = 0;
	 for(auto iter = begin(); iter != end(); iter++)
	 {
	    sum += iter->GetDegree();
	 }
	 return sum / 2;
      }
      inline size_t GetDegree(size_t indexOfNode)
      {
	 assert(HasNode(indexOfNode));
	 return find(indexOfNode)->GetDegree();
      }
      inline size_t GetNumberOfNodes(){return node_list.size();}

   public://edge operations, all of them are abstract virtual function

      virtual bool AddEdge(size_t indexOfHead, size_t indexOfTail, bool reverse=false)=0;

      virtual bool AddEdge(size_t indexOfHead, IndexList& list, bool reverse=false)=0;

      virtual bool AddEdge(iterator& head, size_t indexOfTail, bool reverse = false)=0;

      virtual bool AddEdge(iterator head, IndexList& list, bool reverse=false)=0;

      virtual bool AddEdge(iterator& head, iterator& tail, bool reverse = false) = 0;

      virtual bool RemoveEdge(size_t indexOfHead, size_t indexOfTail, 
			      bool reverse = false)=0;

      virtual bool RemoveEdge(iterator& head, size_t indexOfTail, 
			      bool reverse = false)=0;
      
      virtual bool RemoveEdge(iterator& head, iterator& tail, bool reverse = false)=0;

      virtual bool HasEdge(size_t indexOfHead, size_t indexOfTail, bool reverse=false)=0;
      
      virtual bool HasEdge(iterator& head, size_t indexOfTail, bool reverse = false)=0;

      virtual bool HasEdge(iterator& head, iterator& tail, bool reverse = false)=0;

   public://STL pattern
      
      //void reserve(size_t size){node_list.reserve(size);}

      void clear(){node_list.clear();}
/**
 * @name size - get the number of nodes and edges. If not all these
 * two data are needed, it's better to call GetNumberOf...() instead
 * of this function
 * @return std::pair(number of nodes, number of edges)
 */
      std::pair<size_t, size_t> size()
      {
	 size_t sum = 0;
	 for(auto iter = begin(); iter != end(); iter++)
	 {
	    sum += iter->GetDegree();
	 }
	 return std::make_pair(node_list.size(), sum / 2);
      }

      inline Graph::iterator begin(){return iterator(node_list.begin());}
      
      inline Graph::iterator end() {return iterator(node_list.end());}

      inline Graph::iterator find(size_t index){return iterator(node_list.find(index));}

      inline Graph::iterator operator[](size_t index){return iterator(node_list.find(index));}

      inline bool empty(){return node_list.empty();}
   protected:
      std::unordered_map<size_t, _node> node_list;
   };
//////////////////////////////////////////////////////////////
//////////////////////////

   class UGraph : public Graph
   {
   public:
      typedef std::shared_ptr<UGraph>  pGraph;
/**
 * @name UGraph - default constructor
 * @return int
 */
      UGraph(){}

      UGraph(size_t numberOfNodes):Graph(numberOfNodes){}

      UGraph(UGraph &other)
      {
	 //add node
	 for(auto node = other.begin(); node != other.end(); node++)
	 {
	    AddNode(*node);
	 }
	 //add edge
	 for(auto node = other.begin(); node != other.end(); node++)
	 {
	    for(auto tail = node->begin(); tail != node->end(); tail++)
	    {
	       AddEdge(*node, *tail);
	    }
	 }
      }
/**
 * @name UGraph - destructor
 * @return void
 */
      ~UGraph(){}

   public://implementation of virtual function

      inline virtual bool AddEdge(size_t indexOfHead, size_t indexOfTail, 
				  bool reverse = false)
      {
	 assert(HasNode(indexOfHead));
	 assert(HasNode(indexOfTail));
	 assert(indexOfHead != indexOfTail);
	 
	 node_list[indexOfHead].in_degree.insert(indexOfTail);
	 node_list[indexOfTail].in_degree.insert(indexOfHead);
	 return true;
      }

      inline virtual bool AddEdge(size_t indexOfHead, IndexList& list, bool reverse = false)
      {
	 assert(HasNode(indexOfHead));
	 for(auto tail = list.begin(); tail != list.end(); tail++)
	 {
	    assert(HasNode(*tail));
	    assert(indexOfHead != *tail);
	    
	    node_list[indexOfHead].in_degree.insert(*tail);
	    node_list[*tail].in_degree.insert(indexOfHead);
	 }
	 return true;
      }

      inline virtual bool AddEdge(iterator& head, size_t indexOfTail,
				  bool reverse = false)
      {
	 assert(HasNode(indexOfTail));
	 assert(*head != indexOfTail);
	 
	 head->in_degree.insert(indexOfTail);
	 node_list[indexOfTail].in_degree.insert(*head);
	 return true;
      }

      inline virtual bool AddEdge(iterator head, IndexList& list, bool reverse = false)
      {
	 for(auto tail = list.begin(); tail != list.end(); tail++)
	 {
	    assert(HasNode(*tail));
	    assert(*head != *tail);
	    
	    head->in_degree.insert(*tail);
	    node_list[*tail].in_degree.insert(*head);
	 }
	 return true;
      }

      inline virtual bool AddEdge(iterator& head, iterator& tail,
				  bool reverse = false)
      {
	 assert(head != tail);
	 
	 head->in_degree.insert(*tail);
	 tail->in_degree.insert(*head);

	 return true;
      }


      inline virtual bool RemoveEdge(size_t indexOfHead, size_t indexOfTail, 
				     bool reverse = false)
      {
	 assert(HasNode(indexOfHead));
	 assert(HasNode(indexOfTail));
	 assert(indexOfHead != indexOfTail);
	 
	 node_list[indexOfHead].in_degree.erase(indexOfTail);
	 node_list[indexOfTail].in_degree.erase(indexOfHead);
	 return true;
      }

      inline virtual bool RemoveEdge(iterator& head, size_t indexOfTail, 
				     bool reverse = false)
      {
	 assert(HasNode(indexOfTail));
	 
	 head->in_degree.erase(indexOfTail);
	 node_list[indexOfTail].in_degree.erase(*head);
	 return true;
      }
      
      inline virtual bool RemoveEdge(iterator& head, iterator& tail,
				     bool reverse = false)
      {
	 head->in_degree.erase(*tail);
	 tail->in_degree.erase(*head);

	 return true;
      }

      inline virtual bool HasEdge(size_t indexOfHead, size_t indexOfTail, 
				  bool reverse = false)
      {
	 assert(HasNode(indexOfHead));
	 assert(HasNode(indexOfTail));
	 if(node_list[indexOfHead].in_degree.find(indexOfTail) == 
	    node_list[indexOfHead].in_degree.end())
	    return false;
	 else
	    return true;
      }
      
      inline virtual bool HasEdge(iterator& head, size_t indexOfTail, 
				  bool reverse = false)
      {
	 assert(HasNode(indexOfTail));
	 
	 if(head->in_degree.find(indexOfTail) == 
	    head->in_degree.end())
	    return false;
	 else
	    return true;
      }


      inline virtual bool HasEdge(iterator& head, iterator& tail, 
			   bool reverse = false)
      {
	 if(head->in_degree.find(*tail) == 
	    head->in_degree.end())
	    return false;
	 else
	    return true;
      }
      
   public://special operations

      Matrix GetAdjacencyMatrix()
      {
	 Matrix matrix(GetNumberOfNodes(), GetNumberOfNodes());
	 //write '1'
	 for(auto node = begin(); node != end(); node++)
	 {
	    for(auto other = node->begin(); other != node->end(); other++)
	    {
	       matrix(*node, *other)= 1.0;
	    }
	 }
	 return matrix;
      }

      void CreateFromAdjacencyMatrix(Matrix &matrix)
      {
	 assert(matrix.IsSymmetric());
	 clear();
	 Create(matrix.GetWidth());
	 for(size_t row = 0; row < matrix.GetHeight(); row++)
	 {
	    for(size_t col = row + 1; col < matrix.GetWidth(); col++)
	    {
	       if(abs(matrix(row, col) - 1.0) < 1e-3)
	       {
		  UGraph::AddEdge(row, col);
	       }
	    }
	 }
      }

      Matrix GetLaplacianMatrix()
      {
	 Matrix matrix = GetAdjacencyMatrix();
	 for(size_t i = 0; i < matrix.GetHeight(); i++)
	 {
	    matrix(i, i) = - valarray<double>(matrix.row(i)).sum();
	 }
	 return matrix;
      }
   };

//////////////////////////////////////////////////////////////
//////////////////////////

/** DGraph
 *
 * directed edge graph. The number of edges between any pair of nodes
 * is no more than 2.
 */

   class DGraph : public Graph
   {
   public:
      typedef std::shared_ptr<DGraph>  pGraph;

      DGraph(){}

      DGraph(size_t numberOfNodes):Graph(numberOfNodes){}
      
      ~DGraph(){}

   public://implementation of virtual function

      inline virtual bool AddEdge(size_t indexOfHead, size_t indexOfTail,
				  bool reverse = false)
      {
	 assert(HasNode(indexOfHead));
	 assert(HasNode(indexOfTail));
	 assert(indexOfHead != indexOfTail);
	 if(reverse)
	 {
	    node_list[indexOfHead].out_degree.insert(indexOfTail);
	    node_list[indexOfTail].in_degree.insert(indexOfHead);
	 }
	 else
	 {
	    node_list[indexOfHead].in_degree.insert(indexOfTail);
	    node_list[indexOfTail].out_degree.insert(indexOfHead);
	 }
	 return true;
      }
      
      inline virtual bool AddEdge(size_t indexOfHead, IndexList& list, bool reverse = false)
      {
	 assert(HasNode(indexOfHead));
	
	 if(reverse)
	 {
	    for(auto tail = list.begin(); tail != list.end(); tail++)
	    {
	       assert(HasNode(*tail));
	       assert(indexOfHead != *tail);
	 
	       node_list[indexOfHead].out_degree.insert(*tail);
	       node_list[*tail].in_degree.insert(indexOfHead);
	    }
	 }
	 else
	 {
	    for(auto tail = list.begin(); tail != list.end(); tail++)
	    {
	       assert(HasNode(*tail));
	       assert(indexOfHead != *tail);
	       
	       node_list[indexOfHead].in_degree.insert(*tail);
	       node_list[*tail].out_degree.insert(indexOfHead);
	    }
	 }
	 return true;
      }

      inline virtual bool AddEdge(iterator& head, size_t indexOfTail,
				  bool reverse = false)
      {
	 assert(HasNode(indexOfTail));
	 if(reverse)
	 {
	    head->out_degree.insert(indexOfTail);
	    node_list[indexOfTail].in_degree.insert(*head);
	 }
	 else
	 {
	    head->in_degree.insert(indexOfTail);
	    node_list[indexOfTail].out_degree.insert(*head);
	 }
	 return true;
      }

      inline virtual bool AddEdge(iterator& head, IndexList& list, bool reverse = false)
      {
	 if(reverse)
	 {
	    for(auto tail = list.begin(); tail != list.end(); tail++)
	    {
	        assert(HasNode(*tail));

		head->out_degree.insert(*tail);
		node_list[*tail].in_degree.insert(*head);
	    }
	 }
	 else
	 {
	    for(auto tail = list.begin(); tail != list.end(); tail++)
	    {
	        assert(HasNode(*tail));

		head->in_degree.insert(*tail);
		node_list[*tail].out_degree.insert(*head);
	    }
	 }
	 return true;
      }


      inline virtual bool AddEdge(iterator& head, iterator& tail,
				  bool reverse = false)
      {
	 if(reverse)
	 {
	    head->out_degree.insert(*tail);
	    tail->in_degree.insert(*head);
	 }
	 else
	 {
	    head->in_degree.insert(*tail);
	    tail->out_degree.insert(*head);
	 }
	 return true;
      }


      inline virtual bool RemoveEdge(size_t indexOfHead, size_t indexOfTail, 
				     bool reverse = false)
      {
	 assert(HasNode(indexOfHead));
	 assert(HasNode(indexOfTail));
	 if(reverse)
	 {
	    node_list[indexOfHead].out_degree.erase(indexOfTail);
	    node_list[indexOfTail].in_degree.erase(indexOfTail);
	 }
	 else
	 {
	    node_list[indexOfHead].in_degree.erase(indexOfTail);
	    node_list[indexOfTail].out_degree.erase(indexOfTail);
	 }
	 return true;
      }


      inline virtual bool RemoveEdge(iterator& head, size_t indexOfTail,
				     bool reverse = false)
      {
	 assert(HasNode(indexOfTail));
	 if(reverse)
	 {
	    head->out_degree.erase(indexOfTail);
	    node_list[indexOfTail].in_degree.erase(*head);
	 }
	 else
	 {
	    head->in_degree.erase(indexOfTail);
	    node_list[indexOfTail].out_degree.erase(*head);
	 }
	 return true;
      }

      
      inline virtual bool RemoveEdge(iterator& head, iterator& tail,
				     bool reverse = false)
      {
	 if(reverse)
	 {
	    head->out_degree.erase(*tail);
	    tail->in_degree.erase(*head);
	 }
	 else
	 {
	    head->in_degree.erase(*tail);
	    tail->out_degree.erase(*head);
	 }
	 return true;
      }

      inline virtual bool HasEdge(size_t indexOfHead, size_t indexOfTail, 
			   bool reverse = false)
      {
	 assert(HasNode(indexOfHead));
	 assert(HasNode(indexOfTail));
	 if(reverse)
	 {
	    if(node_list[indexOfHead].out_degree.find(indexOfTail) ==
	       node_list[indexOfHead].out_degree.end())
	       return false;
	    else
	       return true;
	 }
	 else
	 {
	    if(node_list[indexOfHead].in_degree.find(indexOfTail) ==
	       node_list[indexOfHead].in_degree.end())
	       return false;
	    else
	       return true;
	 }
      }

      
      inline virtual bool HasEdge(iterator& head, size_t indexOfTail,
			   bool reverse = false)
      {
	 assert(HasNode(indexOfTail));

	 if(reverse)
	 {
	    if(head->out_degree.find(indexOfTail) == 
	       head->out_degree.end())
	       return false;
	    else
	       return true;
	 }
	 else
	 {
	    if(head->in_degree.find(indexOfTail) == 
	       head->in_degree.end())
	       return false;
	    else
	       return true;
	 }
      }

      virtual bool HasEdge(iterator& head, iterator& tail, 
			   bool reverse = false)
      {
	 if(reverse)
	 {
	    if(head->out_degree.find(*tail) ==
	       head->out_degree.end())
	       return false;
	    else
	       return true;
	 }
	 else
	 {
	    if(head->in_degree.find(*tail) ==
	       head->in_degree.end())
	       return false;
	    else
	       return true;
	 }
      }
   };
////////////////////////////////////////////////////////////////////////
////////////////////

   template<class NodeData = size_t, class EdgeData = double>
   class UNetwork
   {
   public:
      typedef std::shared_ptr<UNetwork> pNetwork;
      typedef std::shared_ptr<NodeData> pNode;
      typedef std::shared_ptr<EdgeData> pEdge;
   public:
/**
 * @name UNetwork - Constructor with specified network topology. The
 * index in network topology must be less  than 2^32.
 * @param graph -  graph 
 * @return UNetwork
 */
      UNetwork(scn::UGraph::pGraph graph)
	 :graph(graph)
      {
	 if(graph->GetNumberOfNodes() != 0)
	 {
	    CreateCirclePosition(0.5, 0.5, 0.5);
	 }
      }

      ~UNetwork(){}
   public:
      
      UGraph::pGraph GetTopology(){return graph;}

      void SetNodeData(UGraph::iterator &node, pNode &data)
      {
	 node_data[*node] = data;
      }

      void SetNodeData(size_t indexOfNode, pNode &data)
      {
	 assert(graph->HasNode(indexOfNode));
	 node_data[indexOfNode] = data;
      }

      pNode GetNodeData(UGraph::iterator &node)
      {
	 return node_data[*node];
      }

      pNode GetNodeData(size_t indexOfNode)
      {
	 assert(graph->HasNode(indexOfNode));
	 return node_data[indexOfNode];
      }
      void SetEdgeData(UGraph::iterator node1, size_t indexOfNode2, pEdge &data)
      {
	 assert(*node1 <= 0xffffffff);
	 assert(indexOfNode2 <= 0xffffffff);
	 assert(graph->HasNode(indexOfNode2));
	 uint64 index;
	 if(*node1 < indexOfNode2)
	 {
	    index = (*node1) << 32;
	    edge_data[index + indexOfNode2] = data;
	 }
	 else
	 {
	    index = indexOfNode2 << 32;
	    edge_data[index + *node1] = data;
	   
	 }
      }

      void SetEdgeData(size_t indexOfNode1, size_t indexOfNode2, pEdge &data)
      {
	 assert(indexOfNode1 <= 0xffffffff);
	 assert(indexOfNode2 <= 0xffffffff);
	 assert(graph->HasNode(indexOfNode1));
	 assert(graph->HasNode(indexOfNode2));

	 uint64 index;
	 if(indexOfNode1 < indexOfNode2)
	 {
	    index = (indexOfNode1) << 32;
	    edge_data[index + indexOfNode2] = data;
	 }
	 else
	 {
	    index = indexOfNode2 << 32;
	    edge_data[index + indexOfNode1] = data;
	   
	 }
      }
      
      
      pEdge GetEdgeData(UGraph::iterator node1, size_t indexOfNode2)
      {
	 assert(*node1 <= 0xffffffff);
	 assert(indexOfNode2 <= 0xffffffff);
	 assert(graph->HasNode(indexOfNode2));
	 uint64 index;
	 if(*node1 < indexOfNode2)
	 {
	    index = (*node1) << 32;
	    return edge_data[index + indexOfNode2];
	 }
	 else
	 {
	    index = indexOfNode2 << 32;
	    return edge_data[index + *node1];
	 }
      }

      void SetNodePosition(size_t indexOfNode, double x, double y, double z)
      {
	 assert(graph->HasNode(indexOfNode));
	 std::array<double, 3> temp;
	 temp[0] = x;
	 temp[1] = y;
	 temp[2] = z;
	 position[indexOfNode] = temp;
      }

      void SetNodePosition(size_t indexOfNode, std::array<double, 3> &position)
      {
	 assert(graph->HasNode(indexOfNode));
	 this->position[indexOfNode] = position;
      }

      std::array<double, 3> GetNodePosition(UGraph::iterator &node)
      {
	 return position[*node];
      }

      std::array<double, 3> GetNodePosition(size_t indexOfNode)
      {
	 assert(graph->HasNode(indexOfNode));
	 return position[indexOfNode];
      }

   protected:
      
      void CreateCirclePosition(double origin_x, double origin_y, double radius)
      {
	 double angle = 0;
	 double delta = 2 * 3.14 / graph->GetNumberOfNodes();
	 
	 for(auto node = graph->begin(); node != graph->end(); node++, angle+=delta)
	 {
	    std::array<double, 3> temp;
	    temp[0] = origin_x + radius * cos(angle);//x
	    temp[1] = origin_y + radius * sin(angle);//y
	    temp[2] = 0.5;
	    SetNodePosition(*node, temp);
	 }
      }
   protected:
      UGraph::pGraph graph;
      std::unordered_map<size_t, std::array<double, 3>> position;
      std::unordered_map<size_t, pNode> node_data;
      std::unordered_map<scn::uint64, pEdge> edge_data;
   };
};



