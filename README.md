# Mapping & Routing - Public Bulletin Boards
Mapping and routing between potential bulletin board locations using Gmaps and Folium

## Description
This project mainly comprises two parts:
* Collecting, processing, and plotting potential bulletin board locations
* Determining optimal routing to maximize bulletin board discovery in a fixed time window

### Collection and Processing
Collection starts with a basic list of potential places that may have a public bulletin board. From here, these places were passed to various mapping APIs to obtain nearby locations that match the description. The use of several different APIs was an attempt to improve results or otherwise limit costs. A total of 3 mapping APIs were experimented with, Google Maps, Bing Maps, and HERE. 

Gmaps was the primary API used throughout the analysis as it consistently produced the best locational query results and already had a [mature python wrapper](https://github.com/googlemaps/google-maps-services-python). The largest drawback is that its free tier is far less generous in comparison with HERE or Bing, which meant extra precautions were needed to avoid wasteful/duplicated queries. The HERE API also seemed to have a better categorization system in place, using sensible hierarchies to describe locations. Had the API documentation been more thorough, a case could be made for its use over Gmaps. 

Folium was chosen to plot the various locations largely due to its simplicity in creating aesthetically pleasing interactive maps. Customizability ultimately became an issue, however, and in the second notebook, plotly was used to plot routing paths between locations.

* Primary notebook: 
  * `flyer_mapping.ipynb` - [[nbviewer]](https://nbviewer.jupyter.org/github/Rypo/flyer-mapping/blob/master/flyer_mapping.ipynb)
* Supplemental Files: 
  * `bing_api.py` - Wrapper functions around Bing Maps API  
  * `here_api.py` - Wrapper functions around HERE API 


### Routing
After collection and processing, we are left with over 2000 potential locations to check. Obviously, it is infeasible to check all locations in any reasonable amount of time. Instead, we will choose a subset that can be traversed within a prescribed amount of time. We also want to maximize the number of bulletin boards we encounter during the journey. Managing these resources can be defined as a constrained optimization problem. 

Specifically, this is a variant of the Traveling Salesman Problem often called the **Orienteering Problem** (OP) or [Vehicle Routing Problem](https://en.wikipedia.org/wiki/Vehicle_routing_problem) with Profits (VRPP). The nomenclature is not entirely consistent across literature, however, and it also has been referred to as the Selective Traveling Salesman Problem, Maximum Collection Problem, and the Bank Robber Problem. Regardless of terminology, the core concept is the same, find a path that maximizes profit while remaining within a fixed budget.

To achieve this, Tsiligirides' approach was implemented using a MIP solver through [cvxpy](https://www.cvxpy.org/)'s interface. 

* Primary notebook: 
  * `flyer_routing.ipynb` - [[nbviewer]](https://nbviewer.jupyter.org/github/Rypo/flyer-mapping/blob/master/flyer_routing.ipynb)
* Supplemental Files: 
  * `orienteering.py` - Class to setup and solve the Orienteering Problem