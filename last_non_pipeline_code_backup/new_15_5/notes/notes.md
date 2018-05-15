



AERIAL IMAGES: GOOGLE:

Difficult properties to Classify:

    02-14-412-020-0000  -> Land
    01-04-100-008-0000  -> House
    02-01-101-003-1033  -> House   (RANGE_INTERPOLATED)
    02-01-201-023-1024, 02-01-201-023-1093  -> House
    02-01-302-077-1037, 02-01-302-077-1063  -> House
    02-01-308-009-0000 -> House
    02-12-213-001-1142 -> House
    
Mislabeled House

    01-23-402-015-0000
    02-01-100-014-0000
    02-06-200-013-0000
    02-09-104-025-0000
    02-09-108-002-0000 , 02-09-108-003-0000, 02-09-108-005-0000   (Interesting because this guy has many properties)
    02-12-103-007-0000
    02-15-202-010-0000,
    02-15-401-038-0000,
    02-17-104-003-0000, 02-17-104-004-0000, 02-17-104-010-0000, 02-17-105-001-0000, 02-17-105-002-0000   (Interesting 5 property)
    02-18-209-012-0000
    02-22-408-001-0000
    
    
Mislabeled Land:

    01-04-401-012-0000
    01-09-204-024-0000
    01-09-400-006-0000
    01-16-101-004-0000
    02-01-400-017-1123
    02-12-100-128-1016   (RANGE_INTERPOLATED)
    02-12-100-128-1199   (RANGE_INTERPOLATED)

    
    
    

Bad Records:

    01-18-101-003-0000
    01-01-109-001-0000


Notes to consider (While selecting Data):

    1. address starting with 0 ... 
        Example:
            0 GROVE AVE 
            0 SUMMIT ST
            0 VACANT PROPERTY 
        fetches the wrong lat-lon and hence the image fetched is wrong.
        
        However, these property are searched correctly when trying out the adress directly in google map. These prperty are basically road (hence have to be land)
        
    2. Sometimes the raw address "555 E, 33rd place" fetches the wrong Lat-Lon, In such a case the loc_type could be "RANGE_INTERPOLATED" intead of Roof top. Hence it is wise to not consider such records.
    
    3. For training Purpose, select only images that have a complete address, i.e address_line, city, state.
    
    

      