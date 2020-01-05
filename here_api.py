import requests

cats = ['leisure-outdoor', 'administrative-areas-building',
        'natural-geographical', 'petrol-station', 'toilet-rest-area']
valid_modes = ['retrieveAddresses','retrieveAreas','retrieveLandmarks','retrieveAll','trackPosition']
"""Minimal wrappers for working with the Here API"""
class Here:

    def __init__(self, app_id, app_code):
        self.app_id = app_id
        self.app_code = app_code

    def category_info(self, category_id):
        b_url = "https://places.cit.api.here.com/"  # "https://places.api.here.com"
        pth = "places/v1/"
        rsc = "/categories/places/"
        url_path = b_url + pth + rsc + f'{category_id}?'
        qparams = {
            'app_id': self.app_id,
            'app_code': self.app_code,
        }
        return requests.get(url_path, params=qparams).json()


    def geocode(self, address):
        b_url = "https://geocoder.api.here.com/"
        pth = "6.2/"
        rsc = "geocode.json?"
        url_path = b_url + pth + rsc
        qparams = {
            'app_id': self.app_id,
            'app_code': self.app_code,
            'searchtext': address
        }
        resp = requests.get(url_path, params=qparams)
        return resp.json()


    def reverse_geocode(self, point, radius=250, ntop=1, mode='retrieveAddresses'):
        b_url = "https://reverse.geocoder.api.here.com/"
        pth = "6.2/"
        rsc = "reversegeocode.json?"
        url_path = b_url + pth + rsc
        qparams = {
            'app_id': self.app_id,
            'app_code': self.app_code,
            'prox': f'{point[0]},{point[1]},{radius}',
            'maxresults': ntop,
            'mode': mode,
        }
        resp = requests.get(url_path, params=qparams)
        return resp.json()


    def nearby_categories(self, point, radius=20, pretty=False):
        b_url = "https://places.cit.api.here.com/"  # "https://places.api.here.com"
        pth = "places/v1/"
        rsc = "categories/places?"
        url_path = b_url + pth + rsc
        in_at = {'in': f'{point[0]},{point[1]};r={radius}'} if radius is not None else {
            'at': f'{point[0]},{point[1]}'}
        qparams = {
            'app_id': self.app_id,
            'app_code': self.app_code,
            **in_at,
            'pretty': str(pretty).lower(),
        }
        resp=requests.get(url_path, params=qparams)
        return resp.json()


    def search(self, query, point):
        b_url = "https://places.cit.api.here.com/"  # "https://places.api.here.com"
        pth = "places/v1/"
        rsc = "discover/search?"
        url_path = b_url + pth + rsc
        qparams = {
            'app_id': self.app_id,
            'app_code': self.app_code,
            'at': f'{point[0]},{point[1]}',
            'q': query,
        }
        resp = requests.get(url_path, params=qparams)
        return resp.json()


    def explore(self, point, radius=None, drilldown=False, size=100, tf='plain', cs='places', cat=cats):
        """drilldown appends additional exploration entries to the query that let you find similar category items"""
        b_url = "https://places.cit.api.here.com/"  # "https://places.api.here.com"
        pth = "places/v1/"
        rsc = "discover/explore?"
        url_path = b_url + pth + rsc
        in_at = {'in': f'{point[0]},{point[1]};r={radius}'} if radius is not None else {
            'at': f'{point[0]},{point[1]}'}
        qparams = {
            'app_id': self.app_id,
            'app_code': self.app_code,
            **in_at,
            'drilldown': str(drilldown).lower(),
            'size': size,
            'tf': tf,
            'cat': ','.join(cat),
            'cs': cs  # crucial parameter, validates that ALL categories for a place returned,
        }
        resp = requests.get(url_path, params=qparams)
        return resp.json()


    def distance(self, origins, destinations, summary_attrs='traveltime,costfactor,distance', mode='fastest;car;traffic:enabled;motorway:0', departure='now'):
        b_url = "https://matrix.route.api.here.com/"
        pth = "routing/7.2/"
        rsc = "calculatematrix.json?"
        url_path = b_url + pth + rsc

        origins = [origins] if isinstance(origins[0], float) else origins
        destinations = [destinations] if len(destinations) < 3 and isinstance(destinations, (tuple,list)) else destinations
        qparams = {
            'app_id': self.app_id,
            'app_code': self.app_code,
            **{f'start{i}': f'{point[0]},{point[1]}' for i, point in enumerate(origins)},
            **{f'destination{i}': f'{point[0]},{point[1]}' for i, point in enumerate(destinations)},
            'mode': mode,
            'depature': departure,
            'summaryAttributes': summary_attrs
        }
        resp = requests.get(url_path, params=qparams)
        return resp.json()

    def parse_explore(self, response):
        resp_df = json_normalize(response['results']['items'],sep='_')
        resp_df = resp_df.drop(columns=['having','icon','openingHours_label'])#'category_type','category_system','category_title','category_href'
        resp_df['alternativeNames'] = resp_df.alternativeNames.dropna().apply(lambda x: [y['name'] for y in x])
        resp_df['categories'] = resp_df['categories'].apply(lambda x: [y.get('id') for y in x])

        if hasattr(resp_df,'tags'): # .get will not suffice due to array truthiness ambiguity
            resp_df['tags'] = resp_df['tags'].dropna().apply(lambda x: [f"{y.get('id')}:{y.get('group')}" for y in x])
        return resp_df