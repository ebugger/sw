/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVDLA_PRIV_AST_H
#define NVDLA_PRIV_AST_H

#include <unordered_set>
#include <set>
#include <vector>
#include <list>
#include <algorithm>
#include <iterator>
#include <string>
#include <sstream>
#include <unordered_map>
#include <stdexcept>

#include "dlaerror.h"

#include "priv/Check.h"
#include "priv/Type.h"
#include "priv/ASTEnums.h"

#include "ErrorMacros.h"

namespace nvdla
{

namespace priv
{

namespace ast
{

enum EdgeDirectionEnum {
    AST_EDGE_DIRECTION_ENUMS(GEN_ENUM)
};
typedef SequenceEnum<EdgeDirectionEnum, NvU8> EdgeDirection;

// #define AST_EDGE_SIDE_ENUMS(GEN_ENUM) \
//     GEN_ENUM(FIRST, 0U)               \
//     GEN_ENUM(SECOND, 1U)              \
//     GEN_ENUM(BOTH, 2U)

// enum EdgeSideEnum {
//     GEN_ENUM(FIRST, 0U) GEN_ENUM(SECOND, 1U) GEN_ENUM(BOTH, 2U)
// };
// enum EdgeSideEnum {
//     FIRST  = 0U,  输出
//     SECOND = 1U,  输入
//     BOTH   = 2U,
// };
enum EdgeSideEnum {
    AST_EDGE_SIDE_ENUMS(GEN_ENUM)
};
//ast::EdgeSide edge_side( is_input ? ast::EdgeSideEnum::SECOND : ast::EdgeSideEnum::FIRST);
typedef SequenceEnum<EdgeSideEnum, NvU8> EdgeSide;


template <class N, class E> class Graph;



template <class G>
class GraphOrdering
{
public:
    typedef typename G::Node Node;
    typedef typename G::Edge Edge;
    typedef typename G::Elem Elem;
    typedef typename G::NodeSequence NodeSequence;
    typedef typename G::EdgeSequence EdgeSequence;
    typedef typename G::ElemSequence ElemSequence;
    typedef typename G::NodeSequenceIterator NodeSequenceIterator;
    typedef typename G::EdgeSequenceIterator EdgeSequenceIterator;
    typedef typename G::ElemSequenceIterator ElemSequenceIterator;

    GraphOrdering(G *graph) : m_graph(graph) { }
    GraphOrdering() : m_graph(0) { }
    virtual ~GraphOrdering() { }

    G *graph() const { return m_graph; }

    void setGraph(G *graph)
    {
        clear();
        m_graph = graph;
    }

    NodeSequenceIterator findNode(Node *n) { return m_node_order.find(n); }
    EdgeSequenceIterator findEdge(Edge *e) { return m_edge_order.find(e); }
    ElemSequenceIterator findElem(Elem e)  { return m_elem_order.find(e); }

    const ElemSequence &elemOrder() const { return m_elem_order; }
    const NodeSequence &nodeOrder() const { return m_node_order; }
    const EdgeSequence &edgeOrder() const { return m_edge_order; }

    virtual NvDlaError generate() = 0;

    virtual void clear()
    {
        m_node_order.clear();
        m_edge_order.clear();
        m_elem_order.clear();
    }

protected:
    G *m_graph;
    NodeSequence m_node_order;
    EdgeSequence m_edge_order;
    ElemSequence m_elem_order;

    NodeSequenceIterator nodeBegin() { return m_node_order.begin(); }
    NodeSequenceIterator nodeEnd()   { return m_node_order.end();   }

    EdgeSequenceIterator edgeBegin() { return m_edge_order.begin(); }
    EdgeSequenceIterator edgeEnd()   { return m_edge_order.end();   }

    ElemSequenceIterator elemBegin() { return m_elem_order.begin(); }
    ElemSequenceIterator elemEnd()   { return m_elem_order.end();   }

};


class GraphTraversalState
{
public:
    //  note: these state values are strictly ordered...
    enum State { e_undiscovered = 0U, e_discovered = 1U, e_finishable = 2U, e_finished = 3U };

    GraphTraversalState() :
        m_e(e_undiscovered), m_discovered(-1), m_finishable(-1), m_finished(-1) { }

    GraphTraversalState(const GraphTraversalState &o) :
        m_e(o.m_e), m_discovered(o.m_discovered),
        m_finishable(o.m_finishable), m_finished(o.m_finished) { }

    virtual ~GraphTraversalState() { }

    void setDiscovered(int t) { m_e = e_discovered; m_discovered = t; }
    void setFinished(int t)   { m_e = e_finished;   m_finished   = t; }
    void setFinishable(int t) { m_e = e_finishable; m_finishable = t; }

    bool isUndiscovered() const { return m_e == e_undiscovered; }
    bool isDiscovered()   const { return m_discovered != -1; }
    bool isFinished()     const { return m_finished   != -1; }
    bool isFinishable()   const { return m_e == e_finishable; }

    State state()     const { return m_e; }
    int discoveryTime()  const { return m_discovered; }
    int finishTime()     const { return m_finished;   }
    int finishableTime() const { return m_finishable; }

    const std::string toString()
    {
        std::stringstream ss;
        const char * state_names[] = { "und", "dis", "fab", "fin" };
        ss << state_names[int(m_e)] << " dis=" << m_discovered << " fab=" << m_finishable << " fin=" << m_finished;
        //ss << "s=" << state_names[int(m_e)] << " dis=" << m_discovered << " fab=" << m_finishable << " fin=" << m_finished;
        return ss.str();
    }

    bool operator==(const GraphTraversalState &rhs) const
    {
        return (m_e == rhs.m_e) && (m_discovered == rhs.m_discovered) &&
            (m_finishable == rhs.m_finishable) && (m_finished == rhs.m_finished);
    }

    bool operator<(const GraphTraversalState &rhs) const
    {
        if ( *this == rhs )
        {
            return false;
        }
        // we know they won't be == now.

        // states: undiscovered < discovered < finishable < finished
        // within each state (i.e. comparing two with same state) use associated time.
        // but, unless absolutely == (all states' times the same) we should use prior
        // states' times as well to determine order of two items in the same state.

        if ( int(m_e) != int(rhs.m_e) )
        {
            // not in the same state just compare which state they're in
            // this isn't really interesting... typically we're comparing
            // states which are all of one type.  but do something reasonable.
            return int(m_e) < int(rhs.m_e);
        }
        // now we know they're both in the same state ( && != each other )
        switch ( m_e )
        {
            case e_discovered:
                return m_discovered < rhs.m_discovered;
                break;

            case e_finishable:
                if ( m_finishable == rhs.m_finishable )
                {
                    return (m_discovered < rhs.m_discovered);
                }
                else
                {
                    return m_finishable < rhs.m_finishable;
                }
                break;

            case e_finished:
                if ( m_finished == rhs.m_finished )
                {
                    if ( m_finishable == rhs.m_finishable )
                    {
                        return m_discovered < rhs.m_discovered;
                    }
                    else
                    {
                        return m_finishable < rhs.m_finishable;
                    }
                }
                else
                {
                    return m_finished < rhs.m_finished;
                }
                break;

            case e_undiscovered:
            default:
                return false; // should be == as there is no time.
        }

        /*not reached*/
        return false;
    }

protected:
    State m_e;
    int m_discovered;  // -1 := not yet discovered
    int m_finishable;  // -1 := not yet finishable
    int m_finished;    // -1 := not yet finished
};


template <class G> class GraphScoreboard;

//G<- graph
template <class G>
class GraphTraversalPointer
{
public:
    typedef typename G::Node Node;
    typedef typename G::Edge Edge;
    typedef typename G::Elem Elem;
    typedef typename G::ElemSequence ElemSequence;
    typedef typename G::NodeSequence NodeSequence;
    typedef typename G::EdgeSequence EdgeSequence;
    typedef typename G::ElemSequenceIterator ElemSequenceIterator;
    typedef typename G::NodeSequenceIterator NodeSequenceIterator;
    typedef typename G::EdgeSequenceIterator EdgeSequenceIterator;

    typedef typename GraphScoreboard<G>::Score Score;
    typedef typename GraphScoreboard<G>::NodeScoresIterator NodeScoresIterator;
    typedef typename GraphScoreboard<G>::EdgeScoresIterator EdgeScoresIterator;
    typedef typename GraphScoreboard<G>::ElemScoresIterator ElemScoresIterator;


    GraphTraversalPointer() : m_graph(0), m_node(0) { }

    GraphTraversalPointer(G *g, Node *n) : m_graph(g), m_node(n) , m_state() { }
    GraphTraversalPointer(const GraphTraversalPointer &o) : m_graph(o.m_graph), m_node(o.m_node) , m_state(o.m_state) { }
    ~GraphTraversalPointer() { }

    GraphTraversalState state() const { return m_state; }
    void setState(GraphTraversalState s) { m_state = s; }

    G *graph() const { return m_graph; }

    Node *node() const { return m_node; }
    void setNode(Node *n) { m_node = n; /* !!! note m_state is stale !!! */ }

#if 0
    Edge *edge() const { return m_edge; }
    void setEdge() { m_edge = e; } /* note m_state is stale */
#endif

    bool operator==(const GraphTraversalPointer<G> &rhs) const
    {
        return (m_node == rhs.m_node) && (m_state == rhs.m_state);
    }

protected:
    G *m_graph; //graph class
    Node *m_node;
    GraphTraversalState m_state;
};

//构造函数里面只初始化了m_graph成graph
template <class G>
class GraphScoreboard
{
public:
    GraphScoreboard(G *g = 0) : m_graph(g) { }
    virtual ~GraphScoreboard() { }

    typedef typename G::Node Node;
    typedef typename G::Edge Edge;
    typedef typename G::Elem Elem;
    typedef typename G::ElemSequence ElemSequence;
    typedef typename G::NodeSequence NodeSequence;
    typedef typename G::EdgeSequence EdgeSequence;
    typedef typename G::ElemSequenceIterator ElemSequenceIterator;
    typedef typename G::NodeSequenceIterator NodeSequenceIterator;
    typedef typename G::EdgeSequenceIterator EdgeSequenceIterator;

    typedef ast::GraphTraversalState Score;
    typedef std::map<Node *, Score, typename G::nodeCompareFn> NodeScores;
    typedef std::map<Edge *, Score, typename G::edgeCompareFn> EdgeScores;
    typedef std::map<Elem, Score, typename G::elemCompareFn>   ElemScores;
    typedef typename NodeScores::iterator NodeScoresIterator;
    typedef typename EdgeScores::iterator EdgeScoresIterator;
    typedef typename ElemScores::iterator ElemScoresIterator;

    NodeScoresIterator nodeBegin() { return m_node_score.begin(); }
    NodeScoresIterator nodeEnd() { return m_node_score.end(); }
    NodeScoresIterator findNode(Node *n) { return m_node_score.find(n); }

    EdgeScoresIterator edgeBegin() { return m_edge_score.begin(); }
    EdgeScoresIterator edgeEnd() { return m_edge_score.end(); }
    EdgeScoresIterator findEdge(Edge *e) { return m_edge_score.find(e); }

    ElemScoresIterator elemBegin() { return m_elem_score.begin(); }
    ElemScoresIterator elemEnd() { return m_elem_score.end(); }
    ElemScoresIterator findElem(Elem e) { return m_elem_score.find(e); }

    std::pair<NodeScoresIterator, bool> insertNode(std::pair<Node *, GraphTraversalState> p) { return m_node_score.insert(p); }
    std::pair<EdgeScoresIterator, bool> insertEdge(std::pair<Edge *, GraphTraversalState> p) { return m_edge_score.insert(p); }
    std::pair<ElemScoresIterator, bool> insertElem(std::pair<Elem,   GraphTraversalState> p) { return m_elem_score.insert(p); }

    void clear() { m_node_score.clear(); m_edge_score.clear(); m_elem_score.clear(); }
    void setGraph(G *g) { m_graph = g; clear(); }

    //
    // fetch means find it and if it doesn't exist create it and mark as discovered
    //
    EdgeScoresIterator fetchEdgeScore(int time, Edge *edge)
    {
        if(edge->originalTensor()) {
            gLogInfo << "\t\t"<< __func__ << " in m_edge_score edge=" << edge->id()<<"/" <<edge->originalTensor()->getName()<< std::endl;            
        }else {
            gLogInfo << "\t\t"<< __func__ << " in m_edge_score edge=" << edge->id()<<" w/o original tensor" << std::endl;            
        }
        EdgeScoresIterator f_i = findEdge(edge);
        if ( f_i == edgeEnd() )
        {
            std::pair<EdgeScoresIterator, bool> inserted =
                insertEdge(std::pair<Edge *, GraphTraversalState>(edge, GraphTraversalState()));
            f_i = inserted.first;
            f_i->second.setDiscovered(time);gLogInfo <<" \t\t\t->.<- not fund ,insert to it with discoverd id:" <<time << std::endl;
        }
        return f_i;
    }


    NodeScoresIterator fetchNodeScore(int time, Node *node)
    {
        gLogInfo <<"\t" <<__func__ << " in m_node_score, node=" << node->id() << "/" <<node->name()<<std::endl;
        NodeScoresIterator f_i = findNode(node);
        if ( f_i == nodeEnd() )
        {
            std::pair<NodeScoresIterator, bool> inserted =
                insertNode(std::pair<Node *, GraphTraversalState>(node, GraphTraversalState()));
            f_i = inserted.first;
            f_i->second.setDiscovered(time); gLogInfo << " \t\t->.< not fund ,insert to it with discoverd id:" <<time << std::endl;
        } //else {gLogInfo << " \tfund it" << std::endl}
        return f_i;
    }

    // the fetch elem score entry points are slightly different
    // because they are derived from already-present (finished)
    // nodes and edges.
    ElemScoresIterator fetchElemScore(Node *node, Score score)
    {
        Elem elem(node, 0);
        ElemScoresIterator f_i = findElem( elem );
        gLogInfo << __func__ << " node=" << node->id() << "with an empty edge Elem(node*, edge*)pair" <<std::endl;
        if ( f_i == elemEnd() )
        {
            gLogInfo <<" \t\t->.<- ElemScore not found, insert into m_elem_score with \"empty\" edge." << std::endl;
            insertElem(std::pair<Elem, Score>(elem, score));
        }
        return f_i;
    }

    ElemScoresIterator fetchElemScore(Edge *edge, Score score) 
    {
        Elem elem(0, edge); //Elem is <node*, edge*> pair
        ElemScoresIterator f_i = findElem( elem );
        gLogInfo << __func__ << "with an empty node Elem(node*, edge*)pair" << std::endl;
        if ( f_i == elemEnd() )
        {
            gLogInfo <<" \t\t->.<- ElemScore not found, insert into m_elem_score with \"empty\" node." << std::endl;
            insertElem(std::pair<Elem, Score>(elem, score));
        }
        return f_i;
    }

    //
    // evaluate edge state based upon upstream nodes.
    // this can result in an edge being *finished*.
    // node evaluation doesn't do that.
    //
    EdgeScoresIterator evaluateEdgeScore(int time, Edge *edge)
    {
        gLogInfo << __func__ << " for edge=" << edge->id() /*<<"/"<<edge->originalTensor()->getName()*/<< std::endl;
        EdgeScoresIterator edge_score_i;
        NodeSequence up_nodes;

        edge_score_i = fetchEdgeScore(time, edge);

        if ( edge_score_i->second.isFinished() )
        {
            gLogInfo << " edge score state is finished so no need upstream and return edge score...." << std::endl;  return edge_score_i;
        }

        up_nodes = m_graph->upstreamNodes(edge);//首次网络input的edge为0 

        bool all_nodes_finished = true; // note: edges w/o upstream nodes are finishable at 0...
        int nodes_finishable_time = 0;
        int nodes_finished_time   = 0;
        gLogInfo << " \t\tget total ↑node numbers: " <<up_nodes.size()<< " so " ;
        for ( size_t un_i = 0, UN_I = up_nodes.size(); un_i != UN_I; ++un_i )
        {
            Node *up_node = up_nodes[un_i];
            NodeScoresIterator node_score_i;
            gLogInfo << "\t";
            node_score_i = fetchNodeScore(time, up_node);

            all_nodes_finished = all_nodes_finished && node_score_i->second.isFinished();
            nodes_finished_time   = std::max<int>(nodes_finished_time, node_score_i->second.finishTime());
            nodes_finishable_time = std::max<int>(nodes_finishable_time, node_score_i->second.finishableTime());
        }

        if ( all_nodes_finished )//如果没有
        {
            edge_score_i->second.setFinishable(nodes_finishable_time); // := same as node finishable
            edge_score_i->second.setFinished(nodes_finished_time); // := same as node finish

            gLogInfo << "\tall up nodes finished, so finish edge=" << edge_score_i->first->id() << " score=" << edge_score_i->second.toString() << std::endl;
            gLogInfo << " \t\t";
            fetchElemScore(edge_score_i->first, edge_score_i->second);
        }

        return edge_score_i;
    }


    //
    // evaluate node state based upon upstream edges.
    // unlike edge evaluation, node evaluation does *not* finish nodes.
    //先找到当前node的score(discoverd time),然后找node 的输入后得到输入的score,再顺着输入继续向上找到node，处理node是否能为finishable.
    NodeScoresIterator evaluateNodeScore(int time, Node *node)
    {
        gLogInfo << __func__ << " node=" << node->id() << " name="<<node->name()<< std::endl;
        NodeScoresIterator node_score_i;

        node_score_i = fetchNodeScore(time, node);

        if ( node_score_i->second.isFinished() )
        {
            gLogInfo << " node score state is finished so no need upstream...." << std::endl;  return node_score_i;
        }

        // input edges
        EdgeSequence up_edges = m_graph->upstreamEdges(node);

        bool all_up_edges_finishable = true; // nodes w/o upstream edges are finishable.向上是可以存在没有edge的情况的
        int all_up_edges_finishable_time = 0;
        gLogInfo << " \tget total ↑edges: "<<up_edges.size() << " and ";
        for (EdgeSequenceIterator ue_i = up_edges.begin(); all_up_edges_finishable && (ue_i != up_edges.end()); ++ue_i )
        {
            EdgeScoresIterator up_edge_score_i;

            up_edge_score_i = evaluateEdgeScore(time, *ue_i);

            all_up_edges_finishable = all_up_edges_finishable && (up_edge_score_i->second.isFinished() || up_edge_score_i->second.isFinishable());
            all_up_edges_finishable_time = std::max<int>(all_up_edges_finishable_time, up_edge_score_i->second.finishableTime());
        }

        if ( all_up_edges_finishable )
        {
            node_score_i->second.setFinishable(all_up_edges_finishable_time + 1); // := last edge completion + 1
            gLogInfo << "\tall_up_edges_finishable so finishable Node=" << node_score_i->first->name() << " score=" << node_score_i->second.toString() << std::endl;
        }

        return node_score_i;
    }


    NodeScoresIterator evaluateScore(GraphTraversalPointer<G> &ptr, int t)
    {
        NodeScoresIterator f_i = evaluateNodeScore(t, ptr.node());
        ptr.setState(f_i->second);
        return f_i;
    }

    //
    // finishes a node and increments time. note: any upstream edges which were finishable
    // but which hadn't been yet take time == t and the node itself takes time = t + 1;
    //are
    NvDlaError finishNode(GraphTraversalPointer<G> &ptr, int t)
    {
        NvDlaError e = NvDlaError_Success;
        gLogInfo << __FUNCTION__<< ":"<<ptr.node()->name()<<" ↑edge to finish if is/are finishable."; 
        EdgeSequence up_edges = m_graph->upstreamEdges(ptr.node());
        for ( EdgeSequenceIterator up_i = up_edges.begin(); up_i != up_edges.end(); ++up_i)
        {
            EdgeScoresIterator edge_score_i = fetchEdgeScore(t, *up_i);
            if ( edge_score_i->second.isFinishable() ) // implies !edge_score_i->second.isFinished() )
            {
                edge_score_i->second.setFinished(t);
                gLogInfo << " at node=" << ptr.node()->id() << " we set from finishable to finish edge=" << (*up_i)->id() << " score=" << edge_score_i->second.toString() << std::endl;
                fetchElemScore(edge_score_i->first, edge_score_i->second);
            }
        }
 
        NodeScoresIterator score_i = fetchNodeScore(t, ptr.node());
        score_i->second.setFinished(t+1);  gLogInfo <<" finished node=" <<score_i->first->name()<<" score=" << score_i->second.toString() <<std::endl;
        ptr.setState(score_i->second);

        fetchElemScore(score_i->first, score_i->second);

        return e;
    }

protected:
    G *m_graph;
    NodeScores m_node_score;  //每种类型的元素对应的状态
    EdgeScores m_edge_score;
    ElemScores m_elem_score;
};




//G<- Graph
template <class G>
class ScoredGraphOrdering
{
public:

    typedef typename G::Node Node;
    typedef typename G::Edge Edge;
    typedef typename G::Elem Elem;

    typedef typename G::NodeSequence NodeSequence;
    typedef typename G::EdgeSequence EdgeSequence;
    typedef typename G::ElemSequence ElemSequence;

    typedef typename G::NodeSequenceIterator NodeSequenceIterator;
    typedef typename G::EdgeSequenceIterator EdgeSequenceIterator;
    typedef typename G::ElemSequenceIterator ElemSequenceIterator;

    typedef typename GraphScoreboard<G>::Score Score;
    typedef typename GraphScoreboard<G>::NodeScores NodeScores;
    typedef typename GraphScoreboard<G>::EdgeScores EdgeScores;
    typedef typename GraphScoreboard<G>::ElemScores ElemScores;

    typedef typename GraphScoreboard<G>::NodeScoresIterator NodeScoresIterator;
    typedef typename GraphScoreboard<G>::EdgeScoresIterator EdgeScoresIterator;
    typedef typename GraphScoreboard<G>::ElemScoresIterator ElemScoresIterator;
    //构造函数初始化了一个graph对象和一个scoreboard对象， scoreboard构造初始化挂载了这个graph
    ScoredGraphOrdering(G *graph = 0) : m_graph(graph), m_scores(graph) { }
    virtual ~ScoredGraphOrdering() { }

    G *graph() const { return m_graph; }
    void setGraph(G *graph)
    {
        clear();
        m_graph = graph;
    }

    GraphScoreboard<G> &scores() { return m_scores; }

    NodeScoresIterator findNodeScore(Node *n) { return m_scores.findNode(n); }
    EdgeScoresIterator findEdgeScore(Edge *e) { return m_scores.findEdge(e); }
    ElemScoresIterator findElemScore(Elem e)  { return m_scores.findElem(e); }

    const std::vector<ElemScoresIterator> &elemScoreOrder() const { return m_elem_score_order; }
    const std::vector<NodeScoresIterator> &nodeScoreOrder() const { return m_node_score_order; }
    const std::vector<EdgeScoresIterator> &edgeScoreOrder() const { return m_edge_score_order; }

    //
    // the real business... generally speaking, does the following:
    //     . seed the scheme with the nodes reachable from the graph's input edges
    //     . while traversal pointers exist
    //         . evaluate nodes at and edges immediately upstream of the pointers
    //         . scrub finished pointers
    //         . find first pointer which is finishable
    //             . finish the node there
    //             . advance pointer downstream
    //                 . add more pointers as needed
    //     . finish off dangling edges
    //     . sort the scoreboard results to provide a linear ordering
    //
    // note that this scheme allows for portions of the graph to go undiscovered
    // (or unfinished) if the input edges to the graph don't cover enough.
    //
    virtual NvDlaError generate() // default is DepthFirstDependencyOrder()
    {
        NvDlaError e = NvDlaError_Success;
        int time = 0;
        typename EdgeSequence::const_iterator ie_i;  //const_iterator vector迭代器的指向不能变，但指向的对象的值可以改变 
        std::list<GraphTraversalPointer<G>> pointer_list;   //在scoreorder创建过程中 临时存在的一个list， 只通过pushback更新

        gLogInfo << "-----------------generating scoreboard----------------------." << std::endl;

        if ( !m_graph ) {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "missing graph");
        }

        clear();

        // for all graph input edges...从输入edge切入
        for ( ie_i = m_graph->inputEdges().begin(); ie_i != m_graph->inputEdges().end(); ++ie_i)
        {
            // this *should* be zero sized, check it. input edge上游不能有node,但必须查找保证不出错， 通过edge的属性中的两个方向的vector查找node
            NodeSequence upstream_nodes = m_graph->upstreamNodes(*ie_i);
            if ( upstream_nodes.size() )
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unexpected source node for network input edge '%s'",
                                     (*ie_i)->id().c_str());
            }
            NodeSequence downstream_nodes = m_graph->downstreamNodes(*ie_i);

            // for all downstream nodes...继续从网络输入edge找到作为输入的下游的node开始,每个node都创建一个GraphTreversePtr
            for ( size_t ni = 0, NI = downstream_nodes.size(); ni != NI; ++ni )
            {
                Node *downstream_node = downstream_nodes[ni];
                //初始化挂载graph和当前edge向下的这个node到这个pointer
                GraphTraversalPointer<G> downstream_pointer(m_graph, downstream_node);
                // don't push more than once as there can be multiple paths to a node from the inputs.
                // also note that is the very first such time this happens and so it's impossible for
                // the GTP<G> pointers to have anything other than "discovered" as their current state.
                // that means the find GTP<G> (downstream_pointer) comparison is really only testing for
                // the presence of the same node.
                //搜索现有， 没有就添加
                if ( pointer_list.end() == std::find(pointer_list.begin(), pointer_list.end(), downstream_pointer) )
                {
                    pointer_list.push_back( downstream_pointer );
                }
            }// end of downstream node 
        }

        while ( pointer_list.size() )
        {
            // helper predicate // a predicate implemented as a class in list remove_if
            struct GraphTraversalPointer_is_Finished
            {
                bool operator() (const GraphTraversalPointer<G> & tptr) { 
                    if(tptr.state().isFinished()) { //当GTP有两个相同的node， 第一个继续往下后(也就是处理完当前这个相同的)，再检查剩下的这个就会发现已经finished了，所以直接删掉
                        gLogInfo <<"  *remove GTP "<<tptr.node()->name()<<" as it's finished " << std::endl;
                        }
                    return tptr.state().isFinished(); }
            };

            //
            // evaluate to find the first which is finishable
            //
            typename std::list<GraphTraversalPointer<G>>::iterator finishable_i;

            finishable_i = pointer_list.end();gLogInfo << "◯->initial finishable iter to past the last element of the GTP list with size:"<< pointer_list.size()<< std::endl;// list::end()是过了最后一个元素的位置，所以引用时没意义的

            for ( typename std::list<GraphTraversalPointer<G>>::iterator p_i = pointer_list.begin();
                  p_i != pointer_list.end();
                  ++p_i)
            {
                NodeScoresIterator check_score_i;
                gLogInfo <<"-------"<<std::endl;
                check_score_i = m_scores.evaluateScore(*p_i, time);

                if ( check_score_i->second.isFinishable() && (finishable_i == pointer_list.end() ) )//下面执行一次后finishable_i指向有意义的元素， 而不是尾部，所以这个分支不会再执行更新finishable_i了。
                {   //相当于当有多个node的时候， 执行第一个node的处理后， 后面的node只跑score，但不处理最终的状态，所以后面的node再后面的处理中会跑多次，也就是临时状态会更新多次。
                    finishable_i = p_i;gLogInfo << "update finishable iter to GTP(<->current node)"<<p_i->node()->name() <<" and ready to finish this node" << std::endl;
                }
            }

            //
            // scrub finished pointers from the list.
            // note: list iterators which aren't removed stay valid.
            //
            pointer_list.remove_if( GraphTraversalPointer_is_Finished() );

            if ( pointer_list.size() == 0 )
            {
                continue; // done.  node一般不会设置成finished， 一般只有edge会被设置成finished.
            }

            //
            // something should be finishable every time through this list 所有node都轮询完毕
            //
            if ( finishable_i == pointer_list.end() )
            {
                sort(); // just so the end result is quasi-legit.  don't ignore failure though.
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "%s is not converging for :%s %s ", __func__, pointer_list.front().node()->id().c_str(), pointer_list.front().node()->name().c_str() );
            }

            GraphTraversalPointer<G> &ptr = *finishable_i;

           //
            // finish and advance downstream 先从上的edge再回来到自身node， 结束后才开始向下寻找edge,然后找到edge下级的node作为下一个循环的开始
            //
            m_scores.finishNode(ptr, time);
            time++; 
            gLogInfo <<"↓edges, also make sure no dangling edges with currecnt processing node" << std::endl;
            //
            // if there's more than one downstream node then add to the pointer list.
            // but add next to this one so stable-ish ordering is maintained.
            //
            NodeSequence advance_nodes;
            EdgeSequence down_edges = m_graph->downstreamEdges(ptr.node());

            // for each downstream edge...
            for ( EdgeSequenceIterator de_i = down_edges.begin(); de_i != down_edges.end(); ++de_i )
            {
                Edge *down_edge = *de_i;
                NodeSequence down_nodes  = m_graph->downstreamNodes(down_edge);

                // make sure any downstream dangling (no output nodes) edges get finished
                m_scores.evaluateEdgeScore(time, down_edge);
                gLogInfo <<"\t↓nodes, insert node numbers:"<<down_nodes.size()<<" to down nodes list ONCE if existed." << std::endl;
                // for each downstream node on the (multi-)edge...如果一个node下面挂了多个edge就要处理这些edges下面的node
                for ( size_t ni = 0, NI = down_nodes.size(); ni != NI; ++ni ) {
                    Node *down_node = down_nodes[ni];

                    // add the nodes to the list only once...
                    if ( advance_nodes.end() == std::find(advance_nodes.begin(), advance_nodes.end(), down_node) )
                    {
                        advance_nodes.push_back(down_node);
                    }
                }//end of收集处理一个edge下挂多个node
            }//end of收集处理一个node下挂的多个edge

            if ( advance_nodes.size() )
            {
#if 1
                gLogInfo << "\tadvance nodes:";
                for ( size_t ai = 0, AI = advance_nodes.size(); ai != AI; ++ai ) {
                    gLogInfo << "[" << advance_nodes[ai]->id() << "] ";
                }
                gLogInfo << std::endl;
#endif

                //
                // the first of these can be followed by our current pointer.  the rest should be inserted.
                // do the insertion in reverse so we can just use finishable_i as the insertion point.
                //finishable_i之前指向原node,现在指向后一个位置,然后一直在这个位置插入元素，而由于是倒序插入，所以最终这个list是个正向的
                int new_pointers = advance_nodes.size();
                typename std::list<GraphTraversalPointer<G>>::iterator insertion_point = ++finishable_i;

                for ( int new_pointer_i = new_pointers - 1; new_pointer_i > 0; --new_pointer_i )
                {
                     gLogInfo << "\t\t insert Multi down node to GTP list with reverse order: "<<advance_nodes[new_pointer_i]->name()<<std::endl;
                    pointer_list.insert( insertion_point, GraphTraversalPointer<G>(m_graph, advance_nodes[new_pointer_i] ) );
                }
                ptr.setNode(advance_nodes[0]); // we'll reevaluate next time through...把下级的状态传递到当前的node(iter指向的),也就是把gtp list的首元素更新，可能还有多余的分支，继续保留
            }//end of 如果收集到下级的node

#if 1
            gLogInfo << "◯<-\t ptr(GTP) list: ";
            for ( typename std::list<GraphTraversalPointer<G>>::iterator ai = pointer_list.begin(); ai != pointer_list.end(); ++ai )
            {
                gLogInfo << "[" << ai->node()->id() <<  "/" << ai->node()->name()<< "] ";
            }
            gLogInfo << std::endl;
#endif
        } // while path pointers exist
      
        sort();
    gLogInfo << "-----------------END generating scoreboard----------------------." << std::endl;
    fail:
        return e;

    }

protected:
    NodeScoresIterator nodeScoreBegin() { return m_scores.nodeBegin(); }
    NodeScoresIterator nodeScoreEnd()   { return m_scores.nodeEnd();   }

    EdgeScoresIterator edgeScoreBegin() { return m_scores.edgeBegin(); }
    EdgeScoresIterator edgeScoreEnd()   { return m_scores.edgeEnd();   }

    ElemScoresIterator elemScoreBegin() { return m_scores.elemBegin(); }
    ElemScoresIterator elemScoreEnd()   { return m_scores.elemEnd();   }

public:

    virtual void clear()
    {
        m_node_score_order.clear();
        m_edge_score_order.clear();
        m_elem_score_order.clear();
        m_scores.clear();
    }


protected:
    G *m_graph;
    GraphScoreboard<G> m_scores;
    std::vector<NodeScoresIterator> m_node_score_order;
    std::vector<EdgeScoresIterator> m_edge_score_order;
    std::vector<ElemScoresIterator> m_elem_score_order;

    struct ElemScoreSorter
    {   
        //uint cnt = 0;
        ElemScoreSorter(GraphScoreboard<G> &scores) : m_use_scores(scores) { }
        bool operator() (ElemScoresIterator i, ElemScoresIterator j)
        {
            NvDlaError e = NvDlaSuccess;

            bool i_lt_j = i->second < j->second;
            bool i_eq_j = i->second == j->second;

            if ( !i->first.first == !i->first.second )
            {
                THROW_ERROR(NvDlaError_InvalidState, "invalid element");
            }

            if ( i_eq_j )
            {
                //if(i->first.second->originalTensor() && j->first.second->originalTensor()){
                    gLogInfo << "type tie-breaker elem i [" << (i->first.first?"node: ":"edge: ") << 
                        (i->first.first?i->first.first->name():(i->first.second->originalTensor()?i->first.second->originalTensor()->getName():"none"))<< i->second.toString() << "] < j[" <<
                        (j->first.first?"node: ":"edge: ") <<(j->first.first?j->first.first->name():(j->first.second->originalTensor()?j->first.second->originalTensor()->getName():"none"))<< j->second.toString() << "] = " << i_lt_j << std::endl;
                // }else {
                //     gLogInfo << "type tie-breaker elem i [" << (i->first.first?"node: ":"edge: ") << 
                //         (i->first.first?i->first.first->name():"None")<< i->second.toString() << "] < j[" <<
                //         (j->first.first?"node: ":"edge: ") <<(j->first.first?j->first.first->name():"None")<< j->second.toString() << "] = " << i_lt_j << std::endl;
                // }
                // nodes before edges if otherwise equal
                if ( i->first.first && j->first.second )
                {
                    i_lt_j = true;
                }
                else if ( i->first.second && j->first.first )
                {
                    i_lt_j = false;
                    // i_eq_j = false;
                }
            }else{
            // gLogInfo << "elem i [" << (i->first.first?"node: ":"edge: ") << i->second.toString() << "] < j[" <<
            //      (j->first.first?"node: ":"edge: ") << j->second.toString() << "] = " << i_lt_j << std::endl;
                // 排序gLogInfo << " elem i [" << (i->first.first?"node: ":"edge: ") << 
                //         (i->first.first?i->first.first->name():i->first.second->originalTensor()->getName())<< i->second.toString() << "] < j[" <<
                //         (j->first.first?"node: ":"edge: ") <<(j->first.first?j->first.first->name():j->first.second->originalTensor()->getName())<< j->second.toString() << "] = " << i_lt_j << std::endl;
            }
            return i_lt_j;
        }
        GraphScoreboard<G> &m_use_scores;
    };

    struct NodeScoreSorter
    {
        //uint cnt = 0;
        NodeScoreSorter(GraphScoreboard<G> &scores) : m_use_scores(scores) { }
        bool operator() (NodeScoresIterator i, NodeScoresIterator j)
        {
            bool i_lt_j = i->second < j->second;
            //排序gLogInfo << "node i "<<i->first->name()<<"[" << i->second.toString() << "] < j[" << j->second.toString() << "]"<<j->first->name() <<" = " << i_lt_j << std::endl;
            return i_lt_j;
        }
        GraphScoreboard<G> &m_use_scores;
    };

    struct EdgeScoreSorter
    {
        //uint cnt = 0;
        EdgeScoreSorter(GraphScoreboard<G> &scores) : m_use_scores(scores) { }
        bool operator() (EdgeScoresIterator i, EdgeScoresIterator j)
        {
            bool i_lt_j = i->second < j->second;
            // 排序<< "edge i "<<i->first->originalTensor()->getName()<<" [" << i->second.toString() << "] < j[" << j->second.toString() << "]"<<j->first->originalTensor()->getName()<<" = " << i_lt_j << std::endl;
            return i_lt_j;
        }
        GraphScoreboard<G> &m_use_scores;
    };

    void sort()
    {
        NodeScoreSorter node_score_sorter(m_scores);
        EdgeScoreSorter edge_score_sorter(m_scores);
        ElemScoreSorter elem_score_sorter(m_scores);

        m_node_score_order.clear(); //std::vector<NodeScoresIterator>
        m_edge_score_order.clear();
        m_elem_score_order.clear();

        NodeScoresIterator node_score_i; //std::map<Node *, Score, typename G::nodeCompareFn>
        EdgeScoresIterator edge_score_i;
        ElemScoresIterator elem_score_i;

        gLogInfo << "sorting graph" << std::endl;
        //注意是copy的iter，而不是原始值(没有解引用)
        for ( node_score_i = m_scores.nodeBegin(); node_score_i != m_scores.nodeEnd(); ++node_score_i ) {
            m_node_score_order.push_back(node_score_i);
        }
#if 0
        //gLogInfo << "m_node_order" << &m_node_order << std::endl;
        gLogInfo << "m_node_order before sort" << std::endl;
        // for ( node_score_i = m_node_score_order.begin(); node_score_i != m_node_score_order.end(); ++node_score_i ) {
        //     gLogInfo << "\t\tsc=[" << node_score_i->second.toString() << "]" << std::endl;
        // }
        for (uint32_t i = 0; i != m_node_score_order.size(); ++i ) {
            gLogInfo << m_node_score_order[i]->first->name()<< "\t\t "<<"sc=[" << m_node_score_order[i]->second.toString() << "]" << std::endl;
        }
#endif
        for ( edge_score_i = m_scores.edgeBegin(); edge_score_i != m_scores.edgeEnd(); ++edge_score_i ) {
            m_edge_score_order.push_back(edge_score_i);
        }

        for ( elem_score_i = m_scores.elemBegin(); elem_score_i != m_scores.elemEnd(); ++elem_score_i ) {
            m_elem_score_order.push_back(elem_score_i);
        }

        std::sort(m_node_score_order.begin(), m_node_score_order.end(), node_score_sorter);
        std::sort(m_edge_score_order.begin(), m_edge_score_order.end(), edge_score_sorter);
        std::sort(m_elem_score_order.begin(), m_elem_score_order.end(), elem_score_sorter);

        gLogInfo << "m_elem_order after sort" << std::endl;
        for ( uint32_t i = 0; i != m_elem_score_order.size(); ++i ) {
            if(m_elem_score_order[i]->first.first) gLogInfo << "N:"<<m_elem_score_order[i]->first.first->name() << "\t\t "<<"sc=[" << m_elem_score_order[i]->second.toString() << "]" << std::endl;
            else if(m_elem_score_order[i]->first.second) 
            {
                std::string c_edge = "Comput Edge";
                gLogInfo <<"E:"<< (m_elem_score_order[i]->first.second->originalTensor()? m_elem_score_order[i]->first.second->originalTensor()->getName() : c_edge )<< "\t\t "<<"sc=[" << m_elem_score_order[i]->second.toString() << "]" << std::endl;
            }
        }
#if 0
        //gLogInfo << "m_node_order" << &m_node_order << std::endl;
        gLogInfo << "m_node_order after sort" << std::endl;
        // for ( node_score_i = m_node_score_order.begin(); node_score_i != m_node_score_order.end(); ++node_score_i ) {
        //     gLogInfo << "\t\tsc=[" << node_score_i->second.toString() << "]" << std::endl;
        // }
        for ( node_score_i = m_scores.nodeBegin(); node_score_i != m_scores.nodeEnd(); ++node_score_i ) {
            gLogInfo << node_score_i->first->name()<< "\t\t "<<"sc=[" << node_score_i->second.toString() << "]" << std::endl;
        }
        gLogInfo << "m_edge_order after sort" << std::endl;
        for ( uint32_t i = 0; i != m_edge_score_order.size(); ++i ) {
            //gLogInfo << m_edge_score_order[i]->first->name()<< std::endl;
            gLogInfo << m_edge_score_order[i]->first->originalTensor()->getName() << "\t\t "<<"sc=[" << m_edge_score_order[i]->second.toString() << "]" << std::endl;
        }
#endif

    }
};

#if 0
template <class G>
class StaticGraphOrdering : public GraphOrdering<G>
{
public:
    StaticGraphOrdering(GraphOrdering<G> *from) : GraphOrdering<G>(from->graph())
    {
        m_node_order = from->nodeOrder();
        m_edge_order = from->edgeOrder();
        m_elem_order = from->elemOrder();
    }

    StaticGraphOrdering(ScoredGraphOrdering<G> *from) : GraphOrdering<G>(from->graph())
    {
        const std::vector<ScoredGraphOrdering<G>::ElemScoresIterator> &elemScoreOrder = from->elemScoreOrder();
        for ( size_t i = 0; i < elemScoreOrder.size(); ++i )
        {
            NvDlaError e = NvDlaSuccess;
            Elem elem = elemScoreOrder[i]->first;
            ScoredGraphOrdering<G>::Score s = elemScoreOrder[i]->second;
            m_elemOrder.push_back(elem);

            if ( !elem.first == !elem.second )
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "illegal element");
            }

            if ( elem.first ) {
                m_nodeOrder.push_back(elem.first);
            }
            else if ( elem.second )
            {
                m_edgeOrder.push_back(elem.second);
            }
        }
    fail:
        return e;

    }

    virtual ~StaticGraphOrdering() { }

    virtual NvDlaError generate()
    {
        NvDlaError e = NvDlaSuccess;
        ORIGINATE_ERROR_FAIL(NvDlaError_InvalidState, "tried to (re)generate a static graph ordering");
    fail:
        return e;
    }

};
#endif

template <class G>
class GraphVisitor
{
public:
    typedef typename G::Node Node;
    typedef typename G::Edge Edge;
    typedef typename G::Elem Elem;
    typedef typename G::NodeSequence NodeSequence;
    typedef typename G::EdgeSequence EdgeSequence;
    typedef typename G::ElemSequence ElemSequence;
    typedef typename G::NodeSequenceIterator NodeSequenceIterator;
    typedef typename G::EdgeSequenceIterator EdgeSequenceIterator;
    typedef typename G::ElemSequenceIterator ElemSequenceIterator;

    GraphVisitor() { }
    virtual ~GraphVisitor() { }

    // note on pure virtual vs. override: i'd prefer to define these and override,
    // but the override keyword doesn't show up until c++11.
    // since we're targetting c++03/08 for misra compliance, and to avoid
    // silently getting the signatures wrong, force pure virtual.
    virtual NvDlaError visitBegin(G *) = 0; // { return NvDlaSuccess; }
    virtual NvDlaError visitEnd(G *, NvDlaError ve) = 0; //   { return ve; }
    virtual NvDlaError visitNode(Node *) = 0; // { return NvDlaSuccess; }
    virtual NvDlaError visitEdge(Edge *) = 0; // { return NvDlaSuccess; }
    virtual NvDlaError visitElem(Elem) = 0; // { return NvDlaSuccess; }

    NvDlaError visitNodes(GraphOrdering<G> *order)
    {
        NvDlaError e = NvDlaSuccess;
        NodeSequenceIterator node_score_i, node_score_end;
        const NodeSequence &nodeOrder = order->nodeOrder();
        PROPAGATE_ERROR_FAIL( visitBegin(order->graph()) );
        for ( size_t i = 0; i < nodeOrder.size(); ++i ) { PROPAGATE_ERROR_FAIL( visitNode(nodeOrder[i]) ); }
    fail:
        e = visitEnd(order->graph(), e);
        return e;
    }

    NvDlaError visitNodes(const NodeSequence &nodeOrder)
    {
        NvDlaError e = NvDlaSuccess;
        if (!nodeOrder.size())
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "visiting an empty node sequence.");
        }
        PROPAGATE_ERROR_FAIL( visitBegin(nodeOrder[0]->graph()) );
        for ( size_t i = 0; i < nodeOrder.size(); ++i ) { PROPAGATE_ERROR_FAIL( visitNode(nodeOrder[i]) ); }
    fail:
        e = visitEnd(nodeOrder[0]->graph(), e);
        return e;
    }

    NvDlaError visitEdges(GraphOrdering<G> *order)
    {
        NvDlaError e = NvDlaSuccess;
        EdgeSequenceIterator edge_score_i, edge_score_end;
        const EdgeSequence &edgeOrder = order->edgeOrder();
        PROPAGATE_ERROR_FAIL( visitBegin(order->graph()) );
        for ( size_t i = 0; i < edgeOrder.size(); ++i ) { PROPAGATE_ERROR_FAIL( visitEdge(edgeOrder[i]) ); }
    fail:
        e = visitEnd(order->graph(), e);
        return e;
    }

    NvDlaError visitElems(GraphOrdering<G> *order)
    {
        NvDlaError e = NvDlaSuccess;
        ElemSequenceIterator elem_score_i, elem_score_end;
        const ElemSequence &elemOrder = order->elemOrder();
        PROPAGATE_ERROR_FAIL( visitBegin(order->graph()) );
        for ( size_t i = 0; i < elemOrder.size(); ++i ) { PROPAGATE_ERROR_FAIL( visitElem(elemOrder[i]) ); }
    fail:
        e = visitEnd(order->graph(), e);
        return e;
    }
};


template <class G>
class ScoredGraphVisitor
{
public:
    typedef typename G::Node Node;
    typedef typename G::Edge Edge;
    typedef typename G::Elem Elem;

    typedef typename GraphScoreboard<G>::Score Score;
    typedef typename GraphScoreboard<G>::NodeScores NodeScores;
    typedef typename GraphScoreboard<G>::EdgeScores EdgeScores;
    typedef typename GraphScoreboard<G>::ElemScores ElemScores;
    typedef typename GraphScoreboard<G>::NodeScoresIterator NodeScoresIterator;
    typedef typename GraphScoreboard<G>::EdgeScoresIterator EdgeScoresIterator;
    typedef typename GraphScoreboard<G>::ElemScoresIterator ElemScoresIterator;

    ScoredGraphVisitor() { }
    virtual ~ScoredGraphVisitor() { }

    // note on pure virtual vs. override: i'd prefer to define these and 'override'.
    // but the override keyword doesn't show up until c++11.
    // since we're targetting c++03/08 for misra compliance and to avoid
    // silently getting the signatures wrong force pure virtual.
    virtual NvDlaError visitBegin(G *, GraphScoreboard< G > &) = 0; // { return NvDlaSuccess; }
    virtual NvDlaError visitEnd(G *, GraphScoreboard< G > &, NvDlaError ve) = 0; //   { return ve; }
    virtual NvDlaError visitNode(Node *, Score) = 0; // { return NvDlaSuccess; }
    virtual NvDlaError visitEdge(Edge *, Score) = 0; // { return NvDlaSuccess; }
    virtual NvDlaError visitElem(Elem ,  Score) = 0; // { return NvDlaSuccess; }

    NvDlaError visitNodes(ScoredGraphOrdering<G> *order)
    {
        NvDlaError e = NvDlaSuccess;
        NodeScoresIterator node_score_i, node_score_end;
        const std::vector<NodeScoresIterator> &nodeOrder = order->nodeScoreOrder();
        PROPAGATE_ERROR_FAIL( visitBegin(order->graph(), order->scores()) );
        for ( size_t i = 0; i < nodeOrder.size(); ++i ) { PROPAGATE_ERROR_FAIL( visitNode(nodeOrder[i]->first, nodeOrder[i]->second) ); }
    fail:
        e = visitEnd(order->graph(), order->scores(), e);
        return e;
    }

    NvDlaError visitEdges(ScoredGraphOrdering<G> *order)
    {
        NvDlaError e = NvDlaSuccess;
        EdgeScoresIterator edge_score_i, edge_score_end;
        const std::vector<EdgeScoresIterator> &edgeOrder = order->edgeScoreOrder();
        PROPAGATE_ERROR_FAIL( visitBegin(order->graph(), order->scores()) );
        for ( size_t i = 0; i < edgeOrder.size(); ++i ) { PROPAGATE_ERROR_FAIL( visitEdge(edgeOrder[i]->first, edgeOrder[i]->second) ); }
    fail:
        e = visitEnd(order->graph(), order->scores(), e);
        return e;
    }

    NvDlaError visitElems(ScoredGraphOrdering<G> *order)
    {
        NvDlaError e = NvDlaSuccess;
        ElemScoresIterator elem_score_i, elem_score_end;
        const std::vector<ElemScoresIterator> &elemOrder = order->elemScoreOrder();
        PROPAGATE_ERROR_FAIL( visitBegin(order->graph(), order->scores()) );
        for ( size_t i = 0; i < elemOrder.size(); ++i ) { PROPAGATE_ERROR_FAIL( visitElem(elemOrder[i]->first, elemOrder[i]->second) ); }
    fail:
        e = visitEnd(order->graph(), order->scores(), e);
        return e;
    }
};


template <class NodeClass, class EdgeClass>
class Graph
{
public:
    typedef ast::Graph<NodeClass, EdgeClass> GraphBase;
    typedef NodeClass Node;
    typedef EdgeClass Edge;
    typedef std::pair<NodeClass *, EdgeClass *> Elem;
    typedef std::vector< Node *> NodeSequence; //Seq都通过vector实现
    typedef std::vector< Edge *> EdgeSequence;
    typedef std::vector< Elem >  ElemSequence;
    typedef typename std::vector< Node *>::iterator NodeSequenceIterator;
    typedef typename std::vector< Edge *>::iterator EdgeSequenceIterator;
    typedef typename std::vector< Elem >::iterator  ElemSequenceIterator;
    /**
     * @brief 比对node在整个网络中的id是否从小到大， 一般在edge的属性中包含输入输出的node vector，这个vector需要排序，现后计算顺序
     * 
     */
    struct nodeCompareFn
    {
        bool operator() (const Node* lhs, const Node* rhs) const
        {
            bool ret = false;
            if (lhs && rhs)
                ret = lhs->uniqueId() < rhs->uniqueId();
            else if (!lhs)
                ret = true;
            else
                ret = false;
            return ret;
        }
    };

    struct edgeCompareFn
    {
        bool operator() (const Edge* lhs, const Edge* rhs) const
        {
            bool ret = false;
            if (lhs && rhs)
                ret = lhs->uniqueId() < rhs->uniqueId();
            else if (!lhs)
                ret = true;
            else
                ret = false;
            return ret;
        }
    };

    struct elemCompareFn
    {
        bool operator() (const Elem& lhs, const Elem& rhs) const
        {
            bool ret = false;
            if (lhs.first && rhs.first)
                ret = lhs.first->uniqueId() < rhs.first->uniqueId();
            else if (lhs.second && rhs.second)
                ret = lhs.second->uniqueId() < rhs.second->uniqueId();
            else if (!lhs.first && rhs.first)
                ret = true;
            else if (!lhs.second && rhs.second)
                ret = true;
            else
                ret = false;
            return ret;
        }
    };
    //包含了set排序的依据函数
    typedef std::set<Node*, nodeCompareFn> NodeSet;
    typedef std::set<Edge*, edgeCompareFn> EdgeSet;
    typedef std::set<Elem, elemCompareFn >  ElemSet;
    typedef typename NodeSet::iterator NodeSetIterator;
    typedef typename EdgeSet::iterator EdgeSetIterator;
    typedef typename ElemSet::iterator  ElemSetIterator;

    typedef std::unordered_set<Node *> NodeUnorderedSet;
    typedef typename NodeUnorderedSet::iterator NodeUnorderedSetIterator;

    Graph() : m_dirty(false) { }
    virtual ~Graph() { }

    virtual Graph *clone() { return new Graph(*this); }

    // any operation which changes the graph will markDirty.
    // client(s) are in charge of noticing this and marking
    // clean after they've taken consideration of the new
    // state.
    bool dirty() const { return m_dirty; }
    virtual void markDirty() { m_dirty = true;  }
    virtual void markClean() { m_dirty = false; }

    // unsorted/ordered graph elements
    const NodeSet & nodes() const { return m_nodes; }
    const EdgeSet & edges() const { return m_edges; }

    // top level graph inputs, outputs
    const EdgeSequence &inputEdges()  const { return m_input_edges;  }
    const EdgeSequence &outputEdges() const { return m_output_edges; }

    Edge* outputEdge(size_t id) {  return (id < m_output_edges.size()) ? m_output_edges[id] : 0; }
    Edge* inputEdge (size_t id) {  return (id < m_input_edges.size())  ? m_input_edges[id]  : 0; }

    void setInputEdges (const EdgeSequence &input)  { m_input_edges  = input;  }
    void setOutputEdges(const EdgeSequence &output) { m_output_edges = output; }

    size_t numNodes()
    {
        return m_nodes.size();
    }

    size_t numEdges()
    {
        return m_edges.size();
    }

    virtual bool insertNode(Node *n)
    {
        NodeSetIterator f = m_nodes.find(n);
        if ( f != m_nodes.end() )
        {
            return false;
        }
        m_nodes.insert(n);

        markDirty();
        return true;
    }

    virtual bool removeNode(Node *node)
    {
        bool found = false;
        NodeAttr *node_attr;

        NodeSetIterator f = m_nodes.find(node);
        EdgeSequenceIterator begin, end;
        if ( f == m_nodes.end() )
        {
            found = false;
            goto done;
        }

        node_attr = lookupNodeAttr(node);

        found = true;

        if ( !node_attr )
        {
            goto done_dirty;
        }
        for (EdgeSide::underlying_type esi = EdgeSideEnum::FIRST, ESI = EdgeSideEnum::SECOND; esi <= ESI; ++esi )
        {
            begin = node_attr->m_edges[esi].begin();
            end   = node_attr->m_edges[esi].end();
            for (EdgeSequenceIterator ei = begin; ei != end; ++ei)
            {
                EdgeAttr *edge_attr = lookupEdgeAttr(*ei);
                if ( edge_attr )
                {
                    removeNodeFromEdge_Internal(edge_attr, node, EdgeSideEnum(esi));
                }
            }
        }

        m_node_attr_map.erase(node);

    done_dirty:
        m_nodes.erase(node);
        markDirty();

    done:
        return found;
    }

    virtual bool insertEdge(Edge *edge)
    {
        EdgeSetIterator f = m_edges.find(edge);
        if ( f != m_edges.end() )
        {
            return false;
        }
        m_edges.insert(edge);
        markDirty();

        return true;
    }

    virtual bool removeEdge(Edge *edge)
    {
        EdgeSetIterator f = m_edges.find(edge);
        if ( f == m_edges.end() )
        {
            return false;
        }
        m_edges.erase(edge);

        markDirty();

        return true;

    }

    // any previously associated nodes with the edge are cleared.
    bool setEdgeNodes(Edge *edge,
                      const NodeSequence &first_nodes,
                      const NodeSequence &second_nodes)
    {
        bool ok = false;
        Node *node;
        NodeAttr  *node_attr;
        EdgeSide::underlying_type edge_side;
        EdgeAttr *edge_attr = fetchEdgeAttr(edge);

        if ( !edge_attr )
        {
            // catch something elswhere... oom failure.
            gLogError << "no edge attr?!?!";
            goto done;
        }

        // detach if previously set
        for ( edge_side = EdgeSideEnum::FIRST; edge_side <= EdgeSideEnum::SECOND; ++edge_side )
        {
            for ( size_t ni = 0, NI = edge_attr->m_nodes[edge_side].size(); ni < NI; ++ni)
            {
                node = edge_attr->m_nodes[edge_side][ni];
                node_attr = lookupNodeAttr(node);
                if ( !node_attr )
                {
                    continue; // must not have been property attached?  not fatal but weird
                    // maybe throw logic error?
                }
                removeEdgeFromNode_Internal(node_attr, edge, edge_side);
            }
        }

        // now attach
        for ( size_t ni = 0, NI = first_nodes.size(); ni < NI; ++ni )
        {
            node = first_nodes[ni];
            node_attr = fetchNodeAttr(node);
            if ( !node_attr )
            {
                // catch oom elsewhere.
                goto done;
            }
            appendEdgeToNode_Internal(node_attr, edge, EdgeSideEnum::FIRST);
        }

        for ( size_t ni = 0, NI = second_nodes.size(); ni < NI; ++ni )
        {
            node = second_nodes[ni];
            node_attr = fetchNodeAttr(node);
            if ( !node_attr )
            {
                // catch oom elsewhere.
                goto done;
            }
            appendEdgeToNode_Internal(node_attr, edge, EdgeSideEnum::SECOND);
        }

        // tbd: there's an optimization to be had above
        // when the set is a simple append or erase of a small subset

        edge_attr->m_nodes[EdgeSideEnum::FIRST]  = first_nodes;
        edge_attr->m_nodes[EdgeSideEnum::SECOND] = second_nodes;

        markDirty();

    done:
        return ok;
    }

    bool adjacentNodes(Node *n1, Node *n2) //任何类型的边都行
    {
        bool res = false;
        NodeAttr *n1_attr = fetchNodeAttr(n1);
        NodeAttr *n2_attr = fetchNodeAttr(n2);

        EdgeSequence n1_input_edges  = n1_attr->m_edges[EdgeSideEnum::SECOND];
        EdgeSequence n1_output_edges = n1_attr->m_edges[EdgeSideEnum::FIRST];
        EdgeSequence n2_input_edges  = n2_attr->m_edges[EdgeSideEnum::SECOND];
        EdgeSequence n2_output_edges = n2_attr->m_edges[EdgeSideEnum::FIRST];

        if (!n1_attr || !n2_attr)
        {
            goto done;
        }

        for (EdgeSequenceIterator n1ie = n1_input_edges.begin(); n1ie != n1_input_edges.end(); ++n1ie)  //如果N1的输入
        {
            if (std::find(n2_output_edges.begin(), n2_output_edges.end(), *n1ie) != n2_output_edges.end()) //包含在N2的输出中
            {
                res = true;
                goto done;
            }
        }

        for (EdgeSequenceIterator n1oe = n1_output_edges.begin(); n1oe != n1_output_edges.end(); ++n1oe) //如果N1的输出
        {
            if (std::find(n2_input_edges.begin(), n2_input_edges.end(), *n1oe) != n2_input_edges.end())//包含在N2的输入
            {
                res = true;
                goto done;
            }
        }

        for (EdgeSequenceIterator n2ie = n2_input_edges.begin(); n2ie != n2_input_edges.end(); ++n2ie)  //如果N2的输入
        {
            if (std::find(n1_output_edges.begin(), n1_output_edges.end(), *n2ie) != n1_output_edges.end()) //包含在N1的输出
            {
                res = true;
                goto done;
            }
        }

        for (EdgeSequenceIterator n2oe = n2_output_edges.begin(); n2oe != n2_output_edges.end(); ++n2oe)  //如果N2的输出
        {
            if (std::find(n1_input_edges.begin(), n1_input_edges.end(), *n2oe) != n1_input_edges.end())  //包含在N1的输入
            {
                res = true;
                goto done;
            }
        }
    done:
        return res;
    }

    NodeSequence edgeNodes(const Edge *edge, EdgeSide side)
    {
        NodeSequence r;
        EdgeAttr *edge_attr = fetchEdgeAttr(edge);
        if ( !edge_attr )
        {
            return r;
        }
        if ( side.e() == EdgeSideEnum::BOTH )
        {
            return r;
        }
        return edge_attr->m_nodes[side.v()];
    }

    size_t numEdgeNodes(const Edge *edge, EdgeSide side) const
    {
        size_t r = 0;
        EdgeAttr *edge_attr = lookupEdgeAttr(edge);
        if ( !edge_attr )
        {
            return 0;
        }
        if (side.e() == EdgeSideEnum::BOTH)
        {
            r = edge_attr->m_nodes[EdgeSideEnum::FIRST].size() +
                edge_attr->m_nodes[EdgeSideEnum::SECOND].size();
        }
        else
        {
            r = edge_attr->m_nodes[side.v()].size();
        }
        return r;
    }


    //当更新完tensor-edge的映射后，把edge和node通过edge的side的方向进行"相互"挂载
    virtual bool appendNodeToEdge(Edge *edge, EdgeSide side, Node *node)
    {   
        //会初始化并挂载到graph的m_edge_attr_map
        EdgeAttr *edge_attr = fetchEdgeAttr(edge);
        NodeAttr *node_attr = fetchNodeAttr(node);
        if ( ! (edge_attr && node_attr) )
        {
            // catch oom elsewhere
            gLogError << "ugh, oom failed" << std::endl;
            return false;
        }
        if ( side.e() == EdgeSideEnum::BOTH )
        {
            appendNodeToEdge_Internal(edge_attr, node, EdgeSideEnum::FIRST);
            appendNodeToEdge_Internal(edge_attr, node, EdgeSideEnum::SECOND);

            appendEdgeToNode_Internal(node_attr, edge, EdgeSideEnum::FIRST);
            appendEdgeToNode_Internal(node_attr, edge, EdgeSideEnum::SECOND);
        }
        else
        {
            //把node加到edge的属性里面
            appendNodeToEdge_Internal(edge_attr, node, side.v());
            //把edge加到node的属性里面
            appendEdgeToNode_Internal(node_attr, edge, side.v());
        }

        markDirty();

        return true;
    }

    virtual bool removeNodeFromEdge(Edge *edge, EdgeSide side, Node *node)
    {
        EdgeAttr *edge_attr = fetchEdgeAttr(edge);
        if( !edge_attr )
        {
            return false;
        }
        if ( side.e() == EdgeSideEnum::BOTH )
        {
            removeNodeFromEdge_Internal(edge_attr, node, EdgeSideEnum::FIRST);
            removeNodeFromEdge_Internal(edge_attr, node, EdgeSideEnum::SECOND);
        }
        else
        {
            removeNodeFromEdge_Internal(edge_attr, node, side.v());
        }

        markDirty();

        return true;
    }

    virtual bool removeEdgeFromNode(Edge *edge, EdgeSide side, Node *node)
    {
        NodeAttr *node_attr = lookupNodeAttr(node);
        if ( !node_attr )
        {
            return false;
        }
        if ( side.e() == EdgeSideEnum::BOTH )
        {
            gLogError << __func__ << " side can't be BOTH " << std::endl;
            return false;
        }
        else
        {
            removeEdgeFromNode_Internal(node_attr, edge, side.v());
        }

        markDirty();

        return true;
    }

    EdgeSequence nodeEdges(const Node *_node, EdgeSide side)
    {
        Node *node = const_cast<Node *>(_node); // no changes...
        NodeAttr *node_attr = lookupNodeAttr(node);
        EdgeSequence all_edges;
        if ( !node_attr)
        {
            return EdgeSequence();
        }
        if (side.e() == EdgeSideEnum::BOTH)
        {
            all_edges.reserve(node_attr->m_edges[EdgeSideEnum::FIRST].size() +
                              node_attr->m_edges[EdgeSideEnum::SECOND].size());
            all_edges.insert(all_edges.end(), node_attr->m_edges[EdgeSideEnum::FIRST].begin(),
                             node_attr->m_edges[EdgeSideEnum::FIRST].end());
            all_edges.insert(all_edges.end(), node_attr->m_edges[EdgeSideEnum::SECOND].begin(),
                             node_attr->m_edges[EdgeSideEnum::SECOND].end());
        }
        else if (side.e() == EdgeSideEnum::FIRST)
        {
            all_edges.reserve(node_attr->m_edges[EdgeSideEnum::FIRST].size());
            all_edges.insert(all_edges.end(), node_attr->m_edges[EdgeSideEnum::FIRST].begin(),
                             node_attr->m_edges[EdgeSideEnum::FIRST].end());
        }
        else if (side.e() == EdgeSideEnum::SECOND)
        {
            all_edges.reserve(node_attr->m_edges[EdgeSideEnum::SECOND].size());
            all_edges.insert(all_edges.end(), node_attr->m_edges[EdgeSideEnum::SECOND].begin(),
                             node_attr->m_edges[EdgeSideEnum::SECOND].end());
        }
        return all_edges;
    }

    size_t numNodeEdges(const Node *node, EdgeSide side)
    {
        size_t r = 0;
        NodeAttr *node_attr = lookupNodeAttr(node);
        if ( !node_attr )
        {
            return 0;
        }
        if (side.e() == EdgeSideEnum::BOTH)
        {
            r = node_attr->m_edges[EdgeSideEnum::FIRST].size() +
                node_attr->m_edges[EdgeSideEnum::SECOND].size();
        }
        else
        {
            r = node_attr->m_edges[side.v()].size();
        }
        return r;
    }

    NodeSequence upstreamNodes(const Edge *edge)
    {
        return edgeNodes(edge, EdgeSideEnum::FIRST);
    }

    NodeSequence downstreamNodes(const Edge *edge)
    {
        return edgeNodes(edge, EdgeSideEnum::SECOND);
    }

    EdgeSequence upstreamEdges(const Node *node)
    {
        return nodeEdges(node, ast::EdgeSideEnum::SECOND);
    }

    EdgeSequence downstreamEdges(const Node *node)
    {
        return nodeEdges(node, ast::EdgeSideEnum::FIRST);
    }

    std::vector<NodeClass *> upstreamNodes(const NodeClass *node)
    {
        std::unordered_set<NodeClass *> unodes; // used to guarantee uniqueness
        std::vector<NodeClass *> allUpstreamNodes = std::vector<NodeClass *>();
        std::vector<EdgeClass *> upstrmEdges = upstreamEdges(node);

        for (typename std::vector<EdgeClass *>::iterator uei = upstrmEdges.begin(); uei != upstrmEdges.end(); ++uei)
        {
            std::vector<NodeClass *> upstrmNodes = upstreamNodes(*uei);
            for ( size_t uni = 0, UNI = upstrmNodes.size(); uni != UNI; ++uni)
            {
                std::pair< typename std::unordered_set<NodeClass *>::iterator, bool> r;
                NodeClass *un = upstrmNodes[uni];
                r = unodes.insert(un);
                if ( r.second == true )
                {
                    allUpstreamNodes.push_back(un);
                }
            }
        }
        return allUpstreamNodes;
    }

    std::vector<NodeClass *> downstreamNodes(const NodeClass *node)
    {
        std::unordered_set<NodeClass *> dnodes; // used to guarantee uniqueness
        std::vector<NodeClass *> allDownstreamNodes;
        std::vector<EdgeClass *> downstrmEdges = downstreamEdges(node);

        for (typename std::vector<EdgeClass *>::iterator dei = downstrmEdges.begin(); dei != downstrmEdges.end(); ++dei)
        {
            std::vector<NodeClass *> downstrmNodes = downstreamNodes(*dei);
            for ( size_t dni = 0, DNI = downstrmNodes.size(); dni != DNI; ++dni )
            {
                std::pair< typename std::unordered_set<NodeClass *>::iterator , bool> r;
                NodeClass *dn = downstrmNodes[dni];
                r = dnodes.insert(dn);
                if ( r.second == true )
                {
                    allDownstreamNodes.push_back(dn);
                }
            }
        }
        return allDownstreamNodes;
    }

protected:
    //
    // use of set is nice here because adding/subtracting is free-ish.
    // below we'll use these as attribute map keys.
    //
    NodeSet m_nodes;
    EdgeSet m_edges;

    EdgeSequence m_input_edges;
    EdgeSequence m_output_edges;

    // we only set/clear here but mark and clean
    // are virtual hooks for more interesting derived behavior.
    bool m_dirty;

    //
    // here we maintain an associative mapping to connectivity and
    // element attribute information.
    ////注意是一个2个元素(分别对应side)的数组， 里面每个元素都是一个edgeSeq的vector，初始化强制vector的大小为1，但后面超出的话会继续分配新的空间2n大小并copy数据过去
    //
    struct NodeAttr
    {   
        
        EdgeSequence m_edges[2]; // for each side, ordered, small-ish, static-ish, vector of edge
        NodeAttr() {
            m_edges[0].reserve(1);
            m_edges[1].reserve(1);
        }
    };
    //注意是一个2个元素(分别对应side)的数组， 每个元素都是一个NodeSeq的vector，初始化强制vector的大小为1，但后面超出的话会继续分配新的空间2n大小并copy数据过去
    struct EdgeAttr
    {   
        NodeSequence m_nodes[2];  // one for each side, ordered, small-ish, static-ish
        EdgeAttr() {
            m_nodes[0].reserve(1);
            m_nodes[1].reserve(1);
        }
    };

    typedef std::unordered_map<const Node *, NodeAttr> NodeAttrMap;
    typedef std::unordered_map<const Edge *, EdgeAttr> EdgeAttrMap;

    NodeAttrMap m_node_attr_map;
    EdgeAttrMap m_edge_attr_map; //edge到edge属性的映射

    inline EdgeAttr *lookupEdgeAttr(const Edge *edge)
    {
        typename EdgeAttrMap::iterator f = m_edge_attr_map.find(edge);
        if ( f == m_edge_attr_map.end() )
        {
            return 0;
        }
        return &f->second;
    }

    inline NodeAttr *lookupNodeAttr(Node *node)
    {
        typename NodeAttrMap::iterator f = m_node_attr_map.find(node);
        if ( f == m_node_attr_map.end() )
        {
            return 0;
        }
        return &f->second;
    }

    //
    // fetch := find || (alloc && insert) (assuming when input is != 0)
    // 在edge到edge属性的映射中查找edge，如果没找到就初始化属性对象然后把这个新的映射在graph中创建好
    EdgeAttr *fetchEdgeAttr(const Edge *edge)
    {
        typename EdgeAttrMap::iterator f;
        EdgeAttr *ret_attr = 0;
        if ( !edge )
        {
            goto done;
        }
        ret_attr = lookupEdgeAttr(edge);
        if ( !ret_attr )
        {
            m_edge_attr_map[edge] = EdgeAttr();
            ret_attr = &m_edge_attr_map[edge];
         }
    done:
        return ret_attr;
    }
    NodeAttr *fetchNodeAttr(Node *node)
    {
        typename NodeAttrMap::iterator f;
        NodeAttr *ret_attr = 0;
        if ( !node )
        {
            goto done;
        }
        ret_attr = lookupNodeAttr(node);
        if ( !ret_attr )
        {
            m_node_attr_map[node] = NodeAttr();
            ret_attr = &m_node_attr_map[node];
        }
    done:
        return ret_attr;
    }

    //将分散的Edge和node通过node属性连接起来，属性里面是两个Edge Seq的vector，两个方式，排序的
    void appendEdgeToNode_Internal(NodeAttr *n_attr, Edge *edge, size_t side)
    {
        EdgeSequenceIterator
            f, begin = n_attr->m_edges[side].begin(), end = n_attr->m_edges[side].end();

        f = std::find( begin, end, edge );
        if ( f == end )
        {
            n_attr->m_edges[side].push_back(edge);
            if (n_attr->m_edges[side].size() > 1)
            {
                std::sort(n_attr->m_edges[side].begin(), n_attr->m_edges[side].end(), edgeCompareFn());
            }
        }
    }
    //将分散的node和edge通过edge属性连接起来，属性里面是两个(方向)Node Seq的vector（等待把node加入进来）， side代表方向 1是输出， 0是输入，如果一个方向连接多个node，需要按照计算顺序排序更新vector
    void appendNodeToEdge_Internal(EdgeAttr *e_attr, Node *node, size_t side)
    {
        NodeSequenceIterator
            f, begin = e_attr->m_nodes[side].begin(), end = e_attr->m_nodes[side].end();

        f = std::find( begin, end, node );
        if ( f == end )
        {
            e_attr->m_nodes[side].push_back(node);
            if (e_attr->m_nodes[side].size() > 1)
            {
                std::sort(e_attr->m_nodes[side].begin(), e_attr->m_nodes[side].end(), nodeCompareFn());
            }
        }
    }

    void removeEdgeFromNode_Internal(NodeAttr *n_attr, Edge *edge, size_t side)
    {
        EdgeSequence new_edges;
        EdgeSequenceIterator
            f, begin = n_attr->m_edges[side].begin(), end = n_attr->m_edges[side].end();
        for ( f = begin; f != end; ++f )
        {
            // lose reference to the sought after edge, don't delete it
            if ( *f != edge )//保留非目标edge
            {
                new_edges.push_back(*f);
            }
        }
        n_attr->m_edges[side] = new_edges;//更新为除目标外的edges
    }

    void removeNodeFromEdge_Internal(EdgeAttr *e_attr, Node *node, size_t side)
    {
        NodeSequence new_nodes;
        NodeSequenceIterator
            f, begin = e_attr->m_nodes[side].begin(), end = e_attr->m_nodes[side].end();
        for ( f = begin; f != end; ++f )
        {
            if ( *f != node )//保留除目标外的node
            {
                new_nodes.push_back(*f);
            }
        }
        e_attr->m_nodes[side] = new_nodes;//更新
    }
};



template <class GraphL, class GraphR>
class GraphMap
{
public:
    GraphMap() : m_gl(0), m_gr(0) {}
    GraphMap(GraphL *l, GraphR *r) : m_gl(l), m_gr(r) {}

    virtual ~GraphMap() { }

    GraphL *graphL()  { return m_gl; }
    void setGraphL(GraphL *l) { clear(); m_gl = l; }
    GraphR *graphR()  { return m_gr; }
    void setGraphR(GraphR *r) { clear(); m_gr = r; }

    typedef typename GraphL::node_class_t l_node_t;
    typedef typename GraphL::edge_class_t l_edge_t;

    typedef typename GraphR::node_class_t r_node_t;
    typedef typename GraphR::edge_class_t r_edge_t;

    typedef std::map<l_node_t *, std::vector<r_node_t *>> lr_node_map_t;
    typedef std::map<r_node_t *, std::vector<l_node_t *>> rl_node_map_t;
    typedef std::map<l_edge_t *, std::vector<r_edge_t *>> lr_edge_map_t;
    typedef std::map<r_edge_t *, std::vector<l_edge_t *>> rl_edge_map_t;

    void clear() {
        m_lr_node_map.clear();
        m_rl_node_map.clear();
        m_lr_edge_map.clear();
        m_rl_edge_map.clear();
    }

    void insertNodeL(const l_node_t *l_node, std::vector<r_node_t *> r_nodes) {
        m_lr_node_map[l_node] = r_nodes;
    }

    void insertEdgeL(const l_edge_t *l_edge, std::vector<r_edge_t *> r_edges) {
        m_lr_edge_map[l_edge] = r_edges;
    }

    void insertNodeR(const r_node_t *r_node, std::vector<l_node_t *> l_nodes) {
        m_rl_node_map[r_node] = l_nodes;
    }

    void insertEdgeR(const r_edge_t *r_edge, std::vector<l_edge_t *> l_edges) {
        m_rl_edge_map[r_edge] = l_edges;
    }

protected:
    GraphL *m_gl;
    GraphR *m_gr;

    lr_node_map_t m_lr_node_map;
    rl_node_map_t m_rl_node_map;
    lr_edge_map_t m_lr_edge_map;
    rl_edge_map_t m_rl_edge_map;

};



// some might not make sense in all contexts
enum PrettyIdFlags
{
    PrettyId_None = 0U,
    PrettyId_Id = 1U,
    PrettyId_ClassName = 2U,
    PrettyId_Name = 4U,
    PrettyId_Type = 8U,
    PrettyId_Default = 0xFU,
    PrettyId_Verbose = 0x10U,
    PrettyId_All = 0xFFU
};




} // nvdla::priv::ast
} // nvdla::priv
} // nvdla

#endif
