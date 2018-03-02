import overpy

api = overpy.Overpass()


# Chicago Bounding BOX : 41.738, -87.803 to 41.999, -87.510


# Extracted GEoJson files for below bounding box in Chicago Region
# 41.738, -87.803, 41.768, -87.510
# 41.768, -87.803, 41.798, -87.510
# 41.798, -87.803, 41.848, -87.510
# 41.848, -87.803, 41.908, -87.510
# 41.908, -87.803, 41.938, -87.510
# 41.938, -87.803, 41.968, -87.510
# 41.968, -87.803, 42.028, -87.510


result = api.query("""
    [out:json];
    (node[building = yes](41.799, -87.706, 41.859, -87.676);
    way[building = yes ](41.799, -87.706, 41.859, -87.676);
    relation[building = yes](41.799, -87.706, 41.859, -87.676););
    (._;>;);
    out body;
    """)#%(str(lower_west_lon), str(lower_west_lat), str(upper_east_lon), str(upper_east_lat))


print (len(result.nodes))
print (len(result.ways))
print (len(result.relations))

for i in result.nodes:
    print (i)
    break
    
# for way in result.ways:
#     print("Name: %s" % way.tags.get("name", "n/a"))
#     print("  Highway: %s" % way.tags.get("highway", "n/a"))
#     print("  Nodes:")
#     for node in way.nodes:
#         print("    Lat: %f, Lon: %f" % (node.lat, node.lon))
#     break