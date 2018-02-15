from urllib import request

from code.config import api_call

print(api_call)

#
# request_string = "http://www.zillow.com/webservice/GetDeepSearchResults.htm?zws-id=X1-ZWz18st70ok00b_4ye7m&address=2114+Bigelow+Ave&citystatezip=Seattle%2C+WA"

#
# with request.urlopen(request_string) as url:
#     s = url.read()
#     print (s)

def get_content(source_url):
    with request.urlopen(source_url) as url:
        s = url.read()
        print (s)
   

# request_url = "https://www.zillow.com/homedetails/2114-Bigelow-Ave-N-Seattle-WA-98109/48749425_zpid/"

request_url = "https://www.zillow.com/homes/for_sale/48749425_zpid/47.68417,-122.280664,47.591636,-122.415419_rect/12_zm/1_fr/"

get_content(request_url)