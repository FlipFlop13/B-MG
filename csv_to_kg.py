import os, sys
import pandas as pd
import numpy as np
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import FOAF , XSD, Namespace
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from hashlib import sha1
import re
import networkx as nx
import matplotlib.pyplot as plt
import glob
import pprint

RDF = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
RDFS = Namespace('http://www.w3.org/2000/01/rdf-schema#')
TOP_LEVEL_DOMAIN = Namespace('https://example.org/')
COUNTRIES =  Namespace('https://example.org/countries/')
SERIES =  Namespace('https://example.org/series/')



def load_csv(path):
    """Loads csv and returns as pandas"""
    path = os.getcwd() + '/' + path
    try:
        encoding_type = "iso-8859-1"
        df = pd.read_csv(path, encoding=encoding_type)
    except UnicodeDecodeError:
        encoding_type = "utf-8"
        df = pd.read_csv(path, encoding=encoding_type)

    return df

def df_to_kg_wbd(df, kg):
    """This function will take every row of a data frame and use the first column as entity, which is the country.
    Then it will take the second column as the series, every series is a subtype of the class series. Every series has
    n-points with a value and a year.
    It will return the updated Knowledge Graph.
    The input data must be of the form provided by the world bank i.e. country, country code, series, series code, year1-yearn
    """
    for index, row in df.iterrows():
        if not "Brazil" in row['Country Name']:
            continue
        country =re.sub(" ", "_", row['Country Name'])
        series = re.sub(" ", "_", row['Series Name'])
        series = re.sub('\(', "-", series)
        series = re.sub("\)", "-", series)

        kg.add((COUNTRIES[country], RDF.type, RDF.country))
        country_series = f"{series}_{country}"
        kg.add((COUNTRIES[country], RDF.hasseries, SERIES[country_series]))
        kg.add((SERIES[country_series], RDF.type, SERIES[series]))
        kg.add((SERIES[series], RDF.type, RDF.series))

        for key, value in row.items():
            year = key.split()[0]#In the year columns there are two year values (int and code)
            try:
                year = int(year)
            except ValueError:
                #This column is not a year entry for  this series
                continue

            if value == "..": #No data point available
                continue

            country_series_point = f"{series}_{country}_{year}"
            kg.add((SERIES[country_series], RDF.haspoint, SERIES[country_series_point]))
            kg.add((SERIES[country_series_point], RDF.hasvalue, Literal(value, datatype=XSD.float)))


    return kg

def df_to_kg_who(df, kg):
    """This function will take every row of a data frame and use the first column as entity, which is the country.
    Then it will take the second column as the series, every series is a subtype of the class series. Every series has
    n-points with a value and a year.
    It will return the updated Knowledge Graph.
    The input data must be of the form provided by the world bank i.e. country, country code, series, series code, year1-yearn
    """

    for index, row in df.iterrows():

        if pd.isna(row['FactValueNumeric']): #Do not add nan
            continue

        country = re.sub(" ", "_", row['Location'])
        series = re.sub(" ", "_", row['Indicator'])
        series = re.sub('\(', "-", series)
        series = re.sub("\)", "-", series)
        year = row['Period']


        #Add the country to the graph
        kg.add((COUNTRIES[country], RDF.type, RDF.country))

        #Add the series, the series of a country with respect to a topic is subclass of the series with respect to that
        #topic which is a subclass of the typee series
        #e.g. Brazil Business Rating series -> Business Rating series -> series
        country_series = f"{series}_{country}"
        kg.add((COUNTRIES[country], RDF.hasseries, SERIES[country_series]))
        kg.add((SERIES[country_series], RDF.type, SERIES[series]))
        kg.add((SERIES[series], RDF.type, RDF.series))


        #Add a point to the series, using the year to distinguish the point
        country_series_point = f"{series}_{country}_{year}"

        value = float(row['FactValueNumeric'])
        kg.add((SERIES[country_series], RDF.haspoint, SERIES[country_series_point]))
        kg.add((SERIES[country_series_point], RDF.hasvalue, Literal(value, datatype=XSD.float)))

    return kg




def main():

    kg = Graph()
    try:
        kg.parse("data/graphs/g_0.ttl")
        pass
    except FileNotFoundError:
        print("RDF not found, initiating a new graph!")
    #     pass
    # for stmt in kg:
    #     pprint.pprint(stmt)

    print(len(kg))
    sys.exit()
    files = glob.glob("data/csv/*.csv")

    for file in files:
        print(f"Adding file {file} to the graph.")
        df = load_csv(file)

        if 'WHO' in file:
            kg = df_to_kg_who(df, kg)
        elif 'WB' in file:
            kg = df_to_kg_wbd(df, kg)


    kg.serialize(destination='data/graphs/g_26_03_16_51.ttl')


if __name__ == "__main__":
    main()
