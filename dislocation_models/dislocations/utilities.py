import numpy as np
from typing import Union
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon, MultiPoint





def slipdip2rake(strike: Union[float, int], dip: Union[float, int], slipvec: Union[float, int]):
    """
    Does exactly the same as the previous function (strikedipsv2rake)
    but has a stupid name!
    Kept here to avoid going through changing scripts

    Program to perform the 'inverse' of slipvec.
    Arguments: strike, dip azimuth of slip vector (all degrees)
    Returns rake (degrees)
    """

    angle = strike - slipvec
    if angle < -180.:
        angle = 360. + angle
    elif angle > 180.:
        angle = angle - 360.

    if angle == 90.:
        rake = 90.
    elif angle == -90.:
        rake = -90.
    else:
        strike_par = np.cos(np.radians(angle))
        strike_perp = np.sin(np.radians(angle)) / np.cos(np.radians(dip))
        rake = np.degrees(np.arctan2(strike_perp, strike_par))
    return rake


def slipvec(strike: Union[float, int], dip: Union[float, int], rake: Union[float, int]):
    """
    Function to find slip vector azimuth from strike, dip,rake (all in degrees)
    Returns azimuth of slip vector in degrees.

    """
    if rake == 180.:
        azimuth = strike - 180.
    else:
        # Separates horizontal component of slip vector
        # into strike-parallel and strike-perpendicular components
        strike_par = np.cos(np.radians(rake))
        strike_perp = np.sin(np.radians(rake)) * np.cos(np.radians(dip))
        # Find angle of slip vector from strike (0 is strike-parallel)
        angle = np.arctan2(strike_perp, strike_par)
        azimuth = strike - np.degrees(angle)
    if azimuth < 0.:
        azimuth = azimuth + 360.
    elif azimuth > 360.:
        azimuth = azimuth - 360.
    return azimuth


def geopandas_polygon_to_gmt(gdf: Union[gpd.GeoDataFrame, gpd.GeoSeries], out_file: str):
    """
    :param gdf: Geodataframe with polygon or multipolygon geometry
    :param out_file: normally has a ".gmt" ending
    :return:
    """
    # Check that writing polygon
    assert all([type(x) in (Polygon, MultiPolygon) for x in gdf.geometry])
    out_id = open(out_file, "w")
    geom_ls = []
    for geom in gdf.geometry:
        if isinstance(geom, Polygon):
            geom_ls.append(geom)
        elif isinstance(geom, MultiPolygon):
            geom_ls += list(geom)
    for geom in geom_ls:
        if isinstance(geom.boundary, LineString):
            x, y = geom.boundary.coords.xy
            out_id.write(">\n")
            for xi, yi in zip(x, y):
                out_id.write("{:.4f} {:.4f}\n".format(xi, yi))
        elif isinstance(geom.boundary, MultiLineString):
            for line in geom.boundary:
                x, y = line.coords.xy
                out_id.write(">\n")
                for xi, yi in zip(x, y):
                    out_id.write("{:.4f} {:.4f}\n".format(xi, yi))
    out_id.close()
    return


def geopandas_linestring_to_gmt(gdf: Union[gpd.GeoDataFrame, gpd.GeoSeries], out_file:str):
    """

    :param gdf: Geodataframe or geoseries with polygon or multipolygon geometry
    :param out_file: normally has a ".gmt" ending
    :return:
    """
    # Check that writing lines
    assert all([type(x) in (LineString, MultiLineString) for x in gdf.geometry])
    out_id = open(out_file, "w")
    geom_ls = []
    for geom in gdf.geometry:
        if isinstance(geom, LineString):
            geom_ls.append(geom)
        elif isinstance(geom, MultiLineString):
            geom_ls += list(geom)
    for geom in geom_ls:
        if isinstance(geom, LineString):
            x, y = geom.coords.xy
            out_id.write(">\n")
            for xi, yi in zip(x, y):
                out_id.write("{:.4f} {:.4f}\n".format(xi, yi))
        elif isinstance(geom, MultiLineString):
            for line in geom:
                x, y = line.coords.xy
                out_id.write(">\n")
                for xi, yi in zip(x, y):
                    out_id.write("{:.4f} {:.4f}\n".format(xi, yi))
    out_id.close()
    return


def geopandas_points_to_gmt(gdf: Union[gpd.GeoDataFrame, gpd.GeoSeries], out_file: str):
    """

    :param gdf: Geodataframe or geoseries with point or multipoint geometry
    :param out_file: normally has a ".gmt" ending
    :return:
    """
    # Check that writing points
    assert all([type(x) in (Point, MultiPoint) for x in gdf.geometry])
    out_id = open(out_file, "w")
    geom_ls = []
    for geom in gdf.geometry:
        if isinstance(geom, Point):
            geom_ls.append(geom)
        elif isinstance(geom, MultiPoint):
            geom_ls += list(geom)
    for geom in geom_ls:
        if isinstance(geom, Point):
            out_id.write("{:.4f} {:.4f}\n".format(geom.x, geom.y))
        elif isinstance(geom, MultiPoint):
            for x, y in geom.xy:
                out_id.write("{:.4f} {:.4f}\n".format(x, y))
    out_id.close()
    return


