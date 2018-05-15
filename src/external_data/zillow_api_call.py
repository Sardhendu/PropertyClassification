from urllib import request

from src.config import api_call

print(api_call)

#

#
# with request.urlopen(request_string) as url:
#     s = url.read()
#     print (s)

def get_content(source_url):
    with request.urlopen(source_url) as url:
        s = url.read()
        print (s)
   


request_url = "https://www.zillow.com/homes/for_sale/48749425_zpid/47.68417,-122.280664,47.591636,-122.415419_rect/12_zm/1_fr/"

get_content(request_url)