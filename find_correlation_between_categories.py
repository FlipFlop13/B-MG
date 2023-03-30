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


RDF = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
RDFS = Namespace('http://www.w3.org/2000/01/rdf-schema#')
TOP_LEVEL_DOMAIN = Namespace('https://example.org/')
COUNTRIES =  Namespace('https://example.org/countries/')
SERIES =  Namespace('https://example.org/series/')



def get_correlation(kg, country, permutation):
    """
    This function will calculate the correlation of a permutation and one country.
    It takes as input the knowledge graph, an country and two series we want to test a correlation on.

    """


    q = f"""
        SELECT ?year ?value
        WHERE {{
            ?country rdf:hasSeries ?y.
            ?y rdf:type ?series.
            ?y rdf:hasPoint ?p.
            ?p rdf:hasValue ?value.
            ?p rdf:year ?year.
        }}
    """
    # Apply the query to the graph and iterate through results
    q_d = {}
    for r in kg.query(q, initBindings={'series': permutation[0], 'country': country}):
        q_d[r['year']] = r['value']
        # print(f"{r['year']}: {r['value']}")

    data_pairs = []
    for r in kg.query(q, initBindings={'series': permutation[1], 'country': country}):
        perm_0 = q_d.get(r['year'], False)  #we must have two points for the same year
        # print(f"{r['year']}: {r['value']}")

        if not perm_0:
            continue
        data_pairs.append([perm_0, r['value']])
    if len(data_pairs) < 2:
        return False

    points = np.array(data_pairs, dtype= float)


    correlation = pearsonr(points[:, 0], points[:, 1])
    # print(f"Correlation coefficient: {correlation[0]:.5f}, p-value: {correlation[1]:.5f}")
    # if correlation[1] < 0.01:
    #     plt.scatter(points[:, 0], points[:, 1])
    #     plt.show()
    return correlation

def find_correlations(kg, classes: tuple = ("Business", "Health")):
    """
    This function will take as input two categories (e.g. Business and Health) and approximate the correlation
    for the combinations of time series in these categories. As running this for every data point and series combination
    we will approximate this value. Second, as we are more interested in series that are correlated, this algorithm will
    look into them in more depth.
    input: Graph RDFLIB.Graph, tuple (class 1, class 2)
    output: n highest correltations as an ordered list (series Business, series Health, r_2)
    """

    #First we get all the series that have the classes.
    class_0_series = []
    class_1_series = []
    for s, p, o in kg.triples((None, RDF.hasSeriesClass, None)):
        if classes[0] in o:
            class_0_series.append(s)
        elif classes[1] in o:
            class_1_series.append(s)

    permutations = [per for per in itertools.product(class_0_series, class_1_series)]
    n_permutations = len(permutations)
    #first we will get an estimate of eachs permutations correlation
    r_scores = [0 for _ in permutations]
    countries = [s for s, p, o in kg.triples((None, None, RDF.country))]

    n_countries = len(countries)
    st = time.time()
    for idx, permutation in enumerate(permutations):
        print(f"\rEstimating correlations {idx}/{n_permutations} ETA:{((n_permutations-idx)*(time.time()-st)/(idx+1)):.2f}s", end = '')
        correlation = 0
        data_entries = 0
        for _ in range(20):
            rand_country = countries[random.randrange(0, n_countries)]
            # rand_country = countries[0]
            ct = get_correlation(kg, rand_country, permutation)
            if not ct or np.isnan(ct[0]):
                continue
            correlation += ct[0]
            data_entries += 1

        try:
            r_scores[idx] = correlation/data_entries
        except ZeroDivisionError:
            r_scores[idx] = 0
    print()
    sorted_indices = np.argsort(r_scores)
    to_save = np.zeros(shape = (len(sorted_indices), 3), dtype=object)
    for i, sorted_idx in enumerate(sorted_indices):
        if r_scores[idx] == 0:
            continue
        print(permutations[sorted_idx][0])
        print(permutations[sorted_idx][1])
        print(f"{r_scores[sorted_idx]:.4f}")
        to_save[i] = [permutations[sorted_idx][0], permutations[sorted_idx][1], r_scores[sorted_idx]]

    np.savetxt("data/correlations_20.csv", to_save, delimiter=",", fmt='%s')

    return



def main():
    graph_path = 'data/graphs/g_30_03_20_55.ttl'

    kg = Graph()
    try:
        print("Loading graph!", end='')
        kg.parse(graph_path)
        print("\rGraph loaded!")
    except FileNotFoundError:
        sys.exit("\rRDF not found, terminating program!")

    find_correlations(kg, ("Business", "Health"))







if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()