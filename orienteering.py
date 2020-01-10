import itertools
import networkx as nx
import cvxpy as cx
import numpy as np
import matplotlib.pyplot as plt

class NotSolvedException(Exception):
    pass

class Orienteering:
    """Orienteering Problem solver
    
    Attempts to solve the Orienteering Problem (Vehicle Routing Problem with Profits),
    using a Mixed-integer programming method through cvxpy.
    The first node in `G` is used as the starting and finishing point in the tour.
    
    Parameters
    ----------
    G : networkx.Graph
        A fully connected graph with node attributes for score, 
        and edge attributes for weight.
    nodes : array-like, optional (default=None)
        Subset of nodes in `G` to use. If None, all nodes will be used.
    n : integer, optional (default=None)
        If not None, a slice of the `n` first nodes will be used as a node subset.
    nodeval : str, (default='value')
        Attribute on every node in `G` that holds the "value" of traveling to that node .
    score_mult : integer, (default=100)
        Multipler applied to node `nodeval` attributes. If solver taking a long time,
        raising or lowering the magnitude of this value can help.
    dwell_time : integer, (default=0)
        Value (in seconds) added to all edges to account for non-travel time-costs.
    nonedge : float, (default=None)
        Value used to fill non-edges in adjecency matrix. Raises a ValueError
        if None and `G` is not a complete graph.
        
    Attributes
    ----------
    cost_matrix_ : np.array, shape (n,n)
        Adjecency matrix of edge weights from `G`.
    score_vector_ : np.array, shape (n,)
        Scores associated with traveling to each node.
    is_solved_ : boolean
        Indicates if problem has been solved.
    tour_ : np.array, shape (nodes_in_tour,)
        Nodes included in the solved tour starting and ending on source.
    costs_ : np.array, shape (nodes_in_tour-1,)
        Array of weights representing travel costs for each step in `tour_`.
    tour_cycle_ : np.array, shape (nodes_in_tour,2)
        pairs of [from,to] returned from networkx.find_cycle.
    problem_data_ : dict
        Contains data returned from and passed to the cvxpy solver.
         solve_time : time (in seconds) for solver to solve problem
         budget : parameter constraint for maximum tour weight.
         cost : total cost of tour. 
         profit : total tour profit / `score_mult`
         x : np.array, shape (n,n) of binary
         prob : cvxpy.Problem
        
    References
    ----------
    Mukhina KD, Visheratin AA, Nasonov D (2019) 
        Orienteering Problem with Functional Profits for multi-source dynamic path construction. 
        PLoS ONE 14(4): e0213777. https://doi.org/10.1371/journal.pone.0213777
    https://github.com/OSUrobotics/Orienteering-Problem-Ebola-Camps
        
    """
    def __init__(self, G, nodes=None, n=None, score_mult=100, nodeval='value', dwell_time=0, nonedge=None):
        self.G = G
        self.nodes = nodes
        self.n = n
        self.score_mult = score_mult
        self.dwell_time = dwell_time
        self._initialize_problem(G=self.G, nodes=self.nodes, n=self.n, 
                                 score_mult=self.score_mult, nodeval=nodeval, 
                                 dwell_time=self.dwell_time, nonedge=nonedge)
        self._initial_attrs = ('G', 'nodes', 'n', 'score_mult', 'dwell_time')
        self._presolve_attrs= ('cost_matrix_', 'score_vector_', 'is_solved_')
        self._postsolve_attrs=('problem_data_', 'tour_', 'costs_', 'tour_cycle_')
                      
    
    def __repr__(self):
        if not self.is_solved_:
            p=[*self._get_params('G','nodes','n','score_mult','dwell_time')]
            return f'unsolved_Orienteering(G={p[0]}, nodes={p[1][:3]}..., n={p[2]}, score_mult={p[3]}, dwell_time={p[4]})'
        else:
            tour, problem_data, costs = self._get_params('tour_','problem_data_','costs_')
            st,cost,profit = problem_data['solve_time'], problem_data['cost'], problem_data['profit']

            return ('Maximum Profit Tour:\n'+' -> '.join(tour.astype(str))+'\n'+
                    'solve_time: {:0.4f} s | tourlen: {} | cost: {:0.2f}({:0.2f}) | profit: {:0.2f}'.format(st, len(tour), cost,costs.sum(), profit))
    
    def _set_params(self, **parameters):
        """Scikit-learn inspired parameter setter"""
        for param, value in parameters.items():
            setattr(self, param, value)
        return self
    
    def _get_params(self, *parameters):
        """Scikit-learn inspired parameter getter"""
        return (getattr(self,param) for param in parameters)
            
    def _initialize_problem(self, G, nodes, n, score_mult, nodeval, dwell_time, nonedge):
        """Should not be called directly. Used as presolve method to setup problem"""
        nodes = nodes if nodes is not None else list(G.nodes)

        if n is not None:  
            n,nodes = n,nodes[:n]
        else:
            n = len(nodes)
        
        _nonedge = nonedge if nonedge is not None else 0.0
        cost_matrix = (nx.to_numpy_array(G, nodes, nonedge=_nonedge)+np.diag([1e6]*n))+dwell_time
        
        if nonedge is None and np.any(cost_matrix-dwell_time == 0):
            raise ValueError('Non-existent (0 Cost) edges found. Specify nonedge value')
        
        score_vector = np.round([G.nodes(nodeval)[n]*score_mult for n in nodes],2)
        score_vector[0]=0
        
        self._set_params(cost_matrix_=cost_matrix, score_vector_=score_vector, nodes=nodes, n=n, is_solved_=False)
    
    def solve(self, budget, solver=None, verbose=False, timeout=60):
        """Use a cvxpy MIP Solver to approximate a tour that maximizes collected profits
        
        Valid solvers may include any of ['CBC','GLPK_MI','CPLEX','ECOS_BB','GUROBI']
        provided that the solver is installed.
        
        For more information on solvers see: 
        https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
        
        Parameters
        ----------
        budget : integer 
            Maximum total tour weight. Total cost of returned route will be <= budget
        solver : str, optional (default=None)
            A valid, installed solver name available to cvxpy
        verbose : boolean, (default=False)
            If true, solver will output information as it runs
        timeout : int, (default=60)
            Time (in seconds) before forcibly stoping the solver 
        """

        cost_matrix, score_vector = self._get_params('cost_matrix_', 'score_vector_')
        
        nnodes=len(score_vector)
        x=cx.Variable((nnodes,nnodes), name='x', boolean=True) # if i -> j is included in tour x_ij is 1 else 0.
        u=cx.Variable(nnodes, name='u') # subtour elimination constraint variables

        cost = cx.trace(cx.matmul(cost_matrix.T,x)) # total cost of the tour
        profit = cx.sum(cx.matmul(x,score_vector))

        nn_ones=np.ones(nnodes)
        # used in place of forloop to clean up constraints
        perm = np.array([*itertools.permutations(np.arange(1,nnodes),2)])
        p0,p1 = perm[:,0], perm[:,1]

        out_con = cx.matmul(x.T,nn_ones) 
        in_con = cx.matmul(x,nn_ones)
        constraints=[cx.sum(x[0,:])==1, # route starts on first node
                     cx.sum(x[:,0])==1, # route ends on first node
                     out_con<=1, # max one outgoing connection
                     in_con<=1, # max one incoming
                     in_con==out_con, # visits = departures
                     cost<=budget, # add the time constraints
                     1 <= u[p0], u[p0] <= nnodes, # subtour elimination
                     u[p0]-u[p1]+1 <= (nnodes-1)*(1-x[p0,p1]), # (Miller-Tucker-Zemlin)
                    ] 
        
        prob=cx.Problem(cx.Maximize(profit),constraints)
        prob.solve(solver=solver, verbose=verbose, TimeLimit=timeout) # TimeLimit may be Gurobi specific

        if prob.status != 'optimal':
            raise cx.SolverError('No feasible solution found, increase budget allotment.')
        
        (x, u), profit = prob.variables(),prob.value
        
        self._set_params(problem_data_={'solve_time': prob.solver_stats.solve_time, 'budget': budget, 'cost': cost.value,
                                        'profit': profit/self.score_mult, 'x': x.value, 'prob' : prob})
        self._set_params(tour_cycle_=self._tour_costs(x.value), is_solved_=True)
        return self
        
    def _tour_costs(self, x):
        """Should not be called directly. Builds and sets tour attributes"""
        # build tour path and tour cost independent from solver.
        cost_matrix, score_vector, nodes = self._get_params('cost_matrix_', 'score_vector_','nodes')
        
        ag = np.argmax(x,axis=1)
        cyc = np.array(nx.find_cycle(nx.Graph([(i,ag[i]) for i in ag]),source=0))
        tour = np.array(nodes)[[0, *cyc[:,1]]] # Initialize with source node.
        costs = cost_matrix[cyc[:,0],cyc[:,1]]
        
        self._set_params(tour_=tour, costs_=costs)
        return cyc
    
    def plot_tour(self, figsize=(8,7), layout=nx.circular_layout, nodesize_mult=750):
        """Plot tour path result from problem solution
        
        Nodes size is determined by its travel value, i.e values in `score_vector_`.
        The source node is colored green, all others are red. 
        
        Parameters
        ----------
        figsize : tuple(int,int), (default=(8,7))
            Adjusts size matplotlib.Figure
        layout : networkx.drawing.layout function (default=networkx.circular_layout)
            Function that determines how to draw the nodes on the graph
        nodesize_mult : integer, (default=750)
            Multipler used to scale nodesize. after min-max scaling nodes
            
        Returns
        -------
        fig,ax : matplotlib Figure,Axis objects
        """
        if not self.is_solved_:
            raise NotSolvedException('Problem not yet solved. Call solve() before proceeding.')
        
        nodes,score_vector,tour_cycle,costs = self._get_params('nodes','score_vector_','tour_cycle_','costs_')
        
        fig,ax = plt.subplots(figsize=figsize)
        color_map=['green']+['red']*(len(nodes)-1)
        
        G=nx.DiGraph()
        G.add_nodes_from([(i+1,{'score':s}) for i,s in enumerate(score_vector)])
        G.add_weighted_edges_from(np.append(tour_cycle+1,np.vstack(costs).astype(np.int),axis=1))
        
        pos = layout(G)
        nodelabs = {a:n for a,n in zip(G.nodes,nodes)}
        nodesize = np.array([v for _,v in G.nodes('score')])
        nodesize = (nodesize-nodesize.min())/nodesize.ptp() # minmax scale
        nodesize[0] = 1
        edgelabs = nx.get_edge_attributes(G,'weight')
        gposax = dict(G=G,pos=pos,ax=ax)

        nx.draw_networkx_nodes(node_size=nodesize*nodesize_mult, node_color=color_map, **gposax)
        nx.draw_networkx_edges(width=1.0, edge_color='k', **gposax)

        nx.draw_networkx_labels(labels=nodelabs, font_size=8, font_color='k', **gposax)
        nx.draw_networkx_edge_labels(edge_labels=edgelabs, **gposax)

        ax.set_title(f'Tour Cost: {costs.sum()}')
        plt.axis('off')
        return fig,ax