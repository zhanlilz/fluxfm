# fluxfm
A library of flux footprint modelling algorithms

## Contacts

* Zhan Li, zhanli AT gfz-potsdam DOT de, zhanli1986 AT gmail DOT com 

---
# Guide for Developers

## Code Style Guide
* __File naming convention__
    * All lower case letters
    * Words separated by underscore "\_"
    * __Use lower cases even a word is an acronym__

    Example: 

    `landsat_view_geo.c`

* __Function naming convention__
    * Start with lower case letters
    * Then separate words by capitalizing initial letters
    * __Use lower cases according to the above rules even a word is an acronym__

    Example:
    
```C
openSensorFile (Dis_landsat_t * sensor)
```
    
* __Variable naming convention__
    * All lower case letters
    * Words separated by underscore "\_"
    * __Use lower cases even a word is an acronym__
    
    Example:
  
```C
double scale_factor;
```

* __Macro, and all the others global variable naming convention__
    * All upper case letters
    * Words separated by underscore "\_"

    Example:

```C
static const char MOD_GRID_NAME[100] = "MOD_Grid_BRDF";

#define N_SRF_BAND (6)

typedef struct {
  Some_type SOME_MEMBER_VARIABLE;
} Dis_landsat_t;
```

* __User-defined type from typedef__
    * Start the first letter in upper case
    * Then all the other letters in lower case
    * Separate words by underscore "\_"
    * Optional: end the type name with "_t" to indicate this is a type name
    * __Use lower cases according to the above rules even a word is an acronym__
    
    Example:
    
```C
typedef enum
{
  MODIS_BRDF = 0,
  VIIRS_BRDF
} Brdf_src_t;
```
    
* __C code indentation__
    * Use the style designated by the following options of the command `indent`
        
        `indent -br -ce -bap --no-tabs`

    Example:
    
```C
for (isc = 0; isc < num_mosaic_scenes; isc++) {
  if (0 != parse_usgs_xml (lndSR[isc]->fileName, &(lndSR[isc]->scene))) {
    fprintf (stderr, "Parse xml failed. XML=%s\n", lndSR[isc]->fileName);
    return FAILURE;
  }
}
```

* __Tabs or spaces__
    * Spaces for indentation in code files
    * Tabs for indentation in makefile
