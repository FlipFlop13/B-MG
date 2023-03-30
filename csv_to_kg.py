import os, sys
import pandas as pd
import numpy as np
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import FOAF, XSD, Namespace
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from hashlib import sha1
import re
import networkx as nx
import matplotlib.pyplot as plt
import glob
import pprint
import time

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
        country =re.sub(" ", "_", row['Country Name'])
        series = re.sub(" ", "_", row['Series Name'])
        series = re.sub('\(', "-", series)
        series = re.sub("\)", "-", series)

        kg.add((COUNTRIES[country], RDF.type, RDF.country))
        country_series = f"{series}_{country}"
        kg.add((COUNTRIES[country], RDF.hasSeries, SERIES[country_series]))
        kg.add((SERIES[country_series], RDF.type, SERIES[series]))
        kg.add((SERIES[series], RDF.type, RDF.series))
        kg.add((SERIES[series], RDF.hasSeriesClass, Literal("Placeholder", datatype=XSD.string)))

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
            kg.add((SERIES[country_series], RDF.hasPoint, SERIES[country_series_point]))
            kg.add((SERIES[country_series_point], RDF.hasValue, Literal(value, datatype=XSD.float)))
            kg.add((SERIES[country_series_point], RDF.year, Literal(year, datatype=XSD.integer)))


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
        kg.add((COUNTRIES[country], RDF.hasSeries, SERIES[country_series]))
        kg.add((SERIES[country_series], RDF.type, SERIES[series]))
        kg.add((SERIES[series], RDF.type, RDF.series))
        kg.add((SERIES[series], RDF.hasSeriesClass, Literal("Placeholder", datatype=XSD.string)))


        #Add a point to the series, using the year to distinguish the point
        country_series_point = f"{series}_{country}_{year}"

        value = float(row['FactValueNumeric'])
        kg.add((SERIES[country_series], RDF.hasPoint, SERIES[country_series_point]))
        kg.add((SERIES[country_series_point], RDF.hasValue, Literal(value, datatype=XSD.float)))
        kg.add((SERIES[country_series_point], RDF.year, Literal(year, datatype=XSD.integer)))

    return kg

def horb(kg):
    """Every series for this project is either a business related series or a health related series.
    The health related series are broader and include things such as education
    """
    with open('data/HorB.txt') as f:
        lines = f.readlines()

    type_dict = {}
    for line in lines:
        line = re.sub('\n', '', line)
        if len(line) == 0:
            continue
        key, type = line.split()
        if not key[-1] == '-':
            print(key)
        type_dict[str(key)] = type

    i = 0
    print(str(key))

    for s, p, o in kg.triples((None, RDF.hasSeriesClass, None)):
        sstring = str(s)
        type = type_dict.get(sstring, "Placeholder")

        kg.remove((s, RDF.hasSeriesClass, None)) #remove the placeholder and add the category
        kg.add((s, RDF.hasSeriesClass, Literal(type, datatype=XSD.string)))


    return kg

def main():
    st = time.time()

    # graph_path = f"data/graphs/g_30_03_20_55.ttl"
    graph_path = "data/graphs/g_temp.ttl"

    kg = Graph()
    try:
        print("Loading graph!", end='')
        kg.parse(graph_path)
        print("\rGraph loaded!")

    except FileNotFoundError:
        print("\rRDF not found, initiating a new graph!")

    files = glob.glob("data/csv/*.csv")

    for idx, file in enumerate(files):
        print(f"Adding file {file} to the graph.")
        df = load_csv(file)
        if 'WHO' in file:
            kg = df_to_kg_who(df, kg)
        elif 'WB' in file:
            kg = df_to_kg_wbd(df, kg)
        # if idx == 5:
        #     break


    kg = horb(kg)
    print("Saving graph!")
    kg.serialize(destination=graph_path)
    print(f"Running time: {(time.time()-st):.4f}")


if __name__ == "__main__":
    main()
