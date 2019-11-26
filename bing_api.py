import requests

"""Minimal wrappers for working with the Bing Maps API"""
class Bing:
    def __init__(self, apikey):
        self.apikey = apikey
        self.b_url = 'https://dev.virtualearth.net/REST/v1'

    @staticmethod
    def template_to_params(template):
        return ', '.join([f"{x}=''" for x in re.findall(r'{(\w+)}', template)])

    def distance(self, destinations, origins, travel_mode='driving', start_time='', time_unit=''):
        """Bing's Distance Matrix API

        References
        ----------
        https://docs.microsoft.com/en-us/bingmaps/rest-services/routes/calculate-a-distance-matrix
        """
        pth = '/Routes/'
        rsc = 'DistanceMatrix?'
        url_path = self.b_url+pth+rsc
        template = url_path+'origins={}&destinations={}&travelMode={}&startTime={}&timeUnit={}&key={}'
        origins = [origins] if isinstance(origins, tuple) else origins
        destinations = [destinations] if isinstance(destinations, tuple) else destinations
            
        forigs = ';'.join([f'{x},{y}' for x,y in origins])
        fdests = ';'.join([f'{x},{y}' for x,y in destinations])
        query = template.format(forigs,fdests,travel_mode,start_time,time_unit,self.apikey)
        resp = requests.get(query)
        return resp.json()['resourceSets']

    def loc_query(self,query, point, max_results=1):
        """Bing's Location Querying/Searching API

        References
        ----------
        https://docs.microsoft.com/en-us/bingmaps/rest-services/locations/local-search
        """
        
        pth = "/LocalSearch/"
        rsc='?'
        url_path = self.b_url+pth+rsc
        params=f'query={query}&userLocation={point[0]},{point[1]}&maxResults={max_results}&key={self.apikey}'
        query = url_path+params
        resp = requests.get(query)
        return resp.json()['resourceSets']

    def loc_recognition(self, point, radius='', n_results='', visit_datetime='', dist_unit='', verbose_names='', entity_types=''):
        """Bing's Location Recognition API
        
        Parameters
        ----------
        point : The coordinates of the location for which you want the entities situated at that location.
        radius : Search radius in kilometers (KM). 
          Search is performed within a circle with the specified radius and centered at the location point.
        n_results : The maximum number of entities returned.
        visit_datetime :. Date and time at which the location is visited, in UTC format.
        dist_unit : Unit for the radius parameter.
        verbose_names : boolean, If false: "admin1" & country names will be
          in their official form (e.g.: “WA” for Washington state in USA and full name
        entity_types : Specifies the entity types returned in the response. Only the specified types will be returned.

        References
        ----------
        https://docs.microsoft.com/en-us/bingmaps/rest-services/locations/location-recognition
        """

        
        pth = "/LocationRecog/"
        url_path = self.b_url+pth
        params= (f"{point[0]},{point[1]}"
                    "?radius={radius}"
                    "&top={n_results}"
                    "&datetime={visit_datetime}"
                    "&distanceunit={dist_unit}"
                    "&verboseplacenames={verbose_names}"
                    "&includeEntityTypes={entity_types}"
                    "&key={self.apikey}")
        
        query = url_path+params
        resp = requests.get(query)
        return resp.json()['resourceSets']
