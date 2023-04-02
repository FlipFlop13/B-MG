from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import FOAF, XSD, Namespace
import itertools
import sys
import random
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import time
import warnings
import math
import multiprocessing as mp
from scipy.stats import pearsonr

if False:#Save to file
    f = open('data/similarity_measurements\correlation_of_prediction_3.txt', 'w')
    sys.stdout = f

RDF = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
RDFS = Namespace('http://www.w3.org/2000/01/rdf-schema#')
TOP_LEVEL_DOMAIN = Namespace('https://example.org/')
COUNTRIES =  Namespace('https://example.org/countries/')
SERIES =  Namespace('https://example.org/series/')

def check_series_and_country_pair(kg, countries,series):
    """
    This function gets two countries and a permutation of two series.
    I returns true if both countries have both series.
    """

    q = f"""
        SELECT ?series
        WHERE {{
            ?country0 rdf:hasSeries ?series0.
            ?country0 rdf:hasSeries ?series1.
            ?country1 rdf:hasSeries ?series0.
            ?country1 rdf:hasSeries ?series1.
            ?series1 rdf:type ?series.
        }}
    """
    series = [r['series0'] for r in kg.query(q, initBindings={'country0': str(countries[0]), 'country1': str(countries[1]),
                                                               'series0': str(series[0]), 'series1': str(series[1])})]
    if len(series) == 1:
        return True

    return False



def print_prediction_relation(kg, countries,series):
    """
    This function gets for each country its most similar pair and get a correlated series that they share and one they dont. Then
    using those three we will infer the third. If it is a test we get 4 series that they all have make a prediciton en check
    similarity.
    """
    q = f"""
            SELECT ?v00 ?v10 ?v01 ?v11
            WHERE {{
                ?country0 rdf:hasSeries ?series_c0_s0.
                ?country0 rdf:hasSeries ?series_c0_s1.
                ?country1 rdf:hasSeries ?series_c1_s0.
                ?country1 rdf:hasSeries ?series_c1_s1.
                ?series_c0_s0 rdf:type ?series0.
                ?series_c1_s0 rdf:type ?series0.
                ?series_c0_s1 rdf:type ?series1.
                ?series_c1_s1 rdf:type ?series1.
                ?series_c0_s0 rdf:hasPoint ?p00.
                ?series_c1_s0 rdf:hasPoint ?p10.
                ?series_c0_s1 rdf:hasPoint ?p01.
                ?series_c1_s1 rdf:hasPoint ?p11.
                ?p00 rdf:hasValue ?v00.
                ?p10 rdf:hasValue ?v10.
                ?p01 rdf:hasValue ?v01.               
                ?p11 rdf:hasValue ?v11.           
                }}
        """
    series = [[r['v00'],r['v10'],r['v01'],r['v11']] for r in kg.query(q, initBindings={'country0': str(countries[0]), 'country1': str(countries[1]), 'series0': str(series[0]), 'series1': str(series[1])})]
    if len(series) == 0:
        return 0

    series = np.array(series, ndim=2)
    numerator = np.sum((series[0,:] - np.mean(series[0,:])) * (series[1,:] - np.mean(series[1,:])))
    denominator = np.sum((series[0,:] - np.mean(series[0,:])) ** 2)
    b1 = numerator/denominator
    b0 = np.mean(series[1,:]) - b1 * np.mean(series[0,:])
    prediction = b0 + b1*series[2]
    correlation = pearsonr(series[1,:],prediction)

    print(f"The correlation this turn was; {correlation};")
    if correlation == np.nan:
        return 0

    return correlation


def main():
    graph_path = 'data/graphs/kg.ttl'

    kg = Graph()
    try:
        print("Loading graph!", end='')
        kg.parse(graph_path)
        print("\rGraph loaded!")
    except FileNotFoundError:
        sys.exit("\rRDF not found, terminating program!")


    sim_0 = np.genfromtxt('data/similarity_measurements/correlations.csv',dtype='object', delimiter=';')
    sim_1 = np.genfromtxt('data/similarity_measurements/country_similarities_top.csv',dtype='object', delimiter=';')
    sim_1_no_c = sim_1[1:,:].astype(float)



    countries = [s for s, p, o in kg.triples((None, None, RDF.country))]
    prediction_correlation = []
    for country_index, similar_to_random_country in enumerate(sim_1_no_c):
        indices = np.argpartition(similar_to_random_country, -1)[-1:]
        for attempt in range(10):
            indicator_0 = [s if str(sim_0[-attempt,0]) in str(s) else None for s, p, o in kg.triples((None, RDF.type, RDF.series))]
            indicator_1 = [s if str(sim_0[-attempt,1]) in str(s) else None for s, p, o in kg.triples((None, RDF.type, RDF.series))]
            t1, t2 = False, False
            for term in indicator_0:
                if term == None:
                    continue
                indicator_0 = term
                t1 = True
            for term in indicator_1:
                if term == None:
                    continue
                indicator_1 = term
                t2 = True
            if not (t1 and t2):
                continue

            if not check_series_and_country_pair(kg, (countries[country_index], countries[indices[0]]),(indicator_0,indicator_1)):
                continue
            try:
                prediction_correlation.append(print_prediction_relation(kg, (countries[country_index], countries[indices[0]]),(indicator_0,indicator_1)))
            except IndexError:
                pass
    print(f"Average correlation between prediction and actual:{np.mean(np.array(prediction_correlation))}")


if __name__ == "__main__":
    main()