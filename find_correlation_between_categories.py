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

RDF = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
RDFS = Namespace('http://www.w3.org/2000/01/rdf-schema#')
TOP_LEVEL_DOMAIN = Namespace('https://example.org/')
COUNTRIES =  Namespace('https://example.org/countries/')
SERIES =  Namespace('https://example.org/series/')
CPU_N = 5
# SIMILARITY_MEASURE = "euclidean"
SIMILARITY_MEASURE = "cosine"

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

def find_series_correlations(kg, classes: tuple = ("Business", "Health")):
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
    correlation_scores_sample = [0 for _ in permutations]
    countries = [s for s, p, o in kg.triples((None, None, RDF.country))]

    n_countries = len(countries)
    st = time.time()
    print("Finding the correlation for interesting pairs, using all the countries' data!")
    #We will first estimate the correlations with a sample of random coutries
    #For each permutation of the series of interest
    for idx, permutation in enumerate(permutations):
        print(f"\rEstimating correlations {idx}/{n_permutations} ETA:{((n_permutations-idx)*(time.time()-st)/(idx+1)):.0f}s", end = '')
        correlation = 0
        data_entries = 0
        #Get a sample of the correlations
        for _ in range(5):
            rand_country = countries[random.randrange(0, n_countries)]
            ct = get_correlation(kg, rand_country, permutation)
            if not ct or np.isnan(ct[0]): #The ct function returns False when the country does not have one of the series
                #and the function returns nan when one of the series has constant value
                continue
            correlation += ct[0]
            data_entries += 1

        try:
            correlation_scores_sample[idx] = correlation/data_entries #Save the average correlation_scores
        except ZeroDivisionError:
            correlation_scores_sample[idx] = 0


    #Repeat the process for all countries where there was evidence of strong correlation from the sample
    correlation_scores_population = []
    for idx, permutation in enumerate(permutations):
        print(
            f"\rEstimating correlations {idx}/{n_permutations} ETA:{((n_permutations - idx) * (time.time() - st) / (idx + 1)):.0f}s",
            end='')
        if abs(correlation_scores_sample[idx]) < 0.8:
            continue
        correlation = 0
        data_entries = 0
        for country in countries:
            ct = get_correlation(kg, country, permutation)
            if not ct or np.isnan(ct[0]):
                continue
            correlation += ct[0]
            data_entries += 1

        try:
            correlation_scores_population.append(correlation / data_entries)
        except ZeroDivisionError:
            correlation_scores_population.append(0)

    print() #Print all the values in ascending correlation order
    sorted_indices = np.argsort(correlation_scores_population)
    to_save = np.zeros(shape = (len(sorted_indices), 3), dtype=object)
    for i, sorted_idx in enumerate(sorted_indices):
        if correlation_scores_sample[idx] == 0:
            continue
        print(permutations[sorted_idx][0])
        print(permutations[sorted_idx][1])
        print(f"{correlation_scores_population[sorted_idx]:.4f}")
        to_save[i] = [permutations[sorted_idx][0], permutations[sorted_idx][1], correlation_scores_population[sorted_idx]]

    np.savetxt("data/similarity_measurements/correlations_20.csv", to_save, delimiter=";", fmt='%s')

    return correlation_scores_population

def get_similarity(kg, permutation, n_series=5):
    """
    Get series for both countries and apply cosine similarity to them.
    Perform this n_series times. Return the average values.
    input: Graph, permutation of two countries, n_series.
    output: similarity value.
    """

    #First get all the series that both countries have
    q = f"""
        SELECT ?series
        WHERE {{
            ?country0 rdf:hasSeries ?seriesC0.
            ?seriesC0 rdf:type ?series.
            ?country1 rdf:hasSeries ?seriesC1.
            ?seriesC1 rdf:type ?series.
        }}
    """
    series = [r['series'] for r in kg.query(q, initBindings={'country0': permutation[0], 'country1': permutation[1]})]
    #get a random sample of those series
    if not n_series == 0: #if n is 0 we will calculate for all series
        try:
            series = random.sample(series, n_series)
        except ValueError:
            n_series = len(series)
            series = random.sample(series, n_series)


    similarity, data_entries = 0, 0
    for idx, serie in enumerate(series):
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
        for r in kg.query(q, initBindings={'series': serie, 'country': permutation[0]}):
            q_d[r['year']] = r['value']

        data_pairs = []
        for r in kg.query(q, initBindings={'series': serie, 'country': permutation[1]}):
            perm_0 = q_d.get(r['year'], False)  # we must have two points for the same year
            if not perm_0:
                continue
            data_pairs.append([perm_0, r['value']])

        if SIMILARITY_MEASURE == "euclidean":
            similarity += math.sqrt(sum(pow(float(a) - float(b), 2) for a, b in data_pairs))
        elif SIMILARITY_MEASURE == "cosine":
            if len(data_pairs) < 2:
                continue
            data_a = np.array([data[0] for data in data_pairs], dtype='float')
            data_b = np.array([data[1] for data in data_pairs], dtype='float')
            similarity += np.dot(data_a, data_b) / (np.linalg.norm(data_a) * np.linalg.norm(data_b))
        data_entries += 1

    if data_entries == 0:
        similarity = 0
    else:
        similarity = similarity / data_entries

    return similarity, n_series


def find_country_similarity(kg):
    """
    This function will return a similarity value between every country pair.
    input: Graph

    """
    countries = [s for s, p, o in kg.triples((None, None, RDF.country))]
    n_countries = len(countries)
    countries_string = np.array([country.split('/')[-1] for country in countries], ndmin=2)
    similarities = np.zeros(shape=(len(countries),len(countries)))
    not_enough_series_found = []
    st = time.time()
    for idx, country in enumerate(countries):
        for i in range(idx+1, n_countries):
            print(
                f"\r{idx}/{n_countries} ETA:{((n_countries - idx) * (time.time() - st) / (idx + 1)):.0f}s; Finding similarity between: {countries_string[0][idx]}<->{countries_string[0][i]}",
                end='')
            n = 5
            similarities[idx][i], n_series = get_similarity(kg, (countries[idx], countries[i]), n_series=n)
            similarities[i][idx] = similarities[idx][i]
            if not n == n_series:
                not_enough_series_found.append([countries_string[0][idx], countries_string[0][i], n_series])

    to_save = np.concatenate((countries_string,similarities))
    np.savetxt("data/similarity_measurements/country_similarities_1_1.csv", to_save, delimiter=";", fmt='%s')

    for not_enough in not_enough_series_found:
        print(not_enough)
    return to_save


def get_similarity_parallel(kg, countries, queue_in, queue_out):
    """
    Get series for both countries and apply cosine similarity to them.
    Perform this n_series times. Return the average values.
    input: Graph, permutation of two countries, n_series.
    output: similarity value.
    """

    while True:
        #First get all the series that both countries have
        x, y, n_series = queue_in.get()
        country0 = countries[x]
        country1 = countries[y]
        q = f"""
            SELECT ?series
            WHERE {{
                ?country0 rdf:hasSeries ?seriesC0.
                ?seriesC0 rdf:type ?series.
                ?country1 rdf:hasSeries ?seriesC1.
                ?seriesC1 rdf:type ?series.
            }}
        """
        series = [r['series'] for r in kg.query(q, initBindings={'country0': country0, 'country1': country1})]
        #get a random sample of those series
        if not n_series == 0: #if n is 0 we will calculate for all series
            try:
                series = random.sample(series, n_series)
            except ValueError:
                n_series = len(series)
                series = random.sample(series, n_series)


        similarity, data_entries = 0, 0
        for idx, serie in enumerate(series):
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
            for r in kg.query(q, initBindings={'series': serie, 'country': country0}):
                q_d[r['year']] = r['value']

            data_pairs = []
            for r in kg.query(q, initBindings={'series': serie, 'country': country1}):
                perm_0 = q_d.get(r['year'], False)  # we must have two points for the same year
                if not perm_0:
                    continue
                data_pairs.append([perm_0, r['value']])
            if SIMILARITY_MEASURE == "euclidean":
                similarity += math.sqrt(sum(pow(float(a)-float(b),2) for a, b in data_pairs))
            elif SIMILARITY_MEASURE == "cosine":
                if len(data_pairs) < 2:
                    continue
                data_a = np.array([data[0] for data in data_pairs], dtype='float')
                data_b = np.array([data[1] for data in data_pairs], dtype='float')
                if np.linalg.norm(data_a) == 0 or np.linalg.norm(data_b) == 0:
                    continue
                similarity += np.dot(data_a,data_b)/(np.linalg.norm(data_a)*np.linalg.norm(data_b))

            data_entries += 1
        if data_entries == 0:
            similarity = -1
        else:
            similarity = similarity / data_entries

        queue_out.put([x, y, similarity, n_series])

        if queue_in.empty():
            queue_out.put(False)
            return


def find_country_similarity_parallel(kg):
    """
    This function will return a similarity value between every country pair.
    input: Graph

    """


    countries = [s for s, p, o in kg.triples((None, None, RDF.country))]
    n_countries = len(countries)
    countries_string = np.array([country.split('/')[-1] for country in countries], ndmin=2)
    similarities = np.zeros(shape=(len(countries),len(countries)))
    n_series = np.zeros(shape=(len(countries),len(countries)))

    queues_in = [mp.Queue() for _ in range(CPU_N)]
    queues_out = [mp.Queue() for _ in range(CPU_N)]
    processess = [mp.Process(target=get_similarity_parallel, args=(kg, countries, queues_in[i], queues_out[i])) for i in
                  range(CPU_N)]
    n_runs = 0
    for idx, country in enumerate(countries):
        process_n = idx % CPU_N
        for i in range(idx+1, n_countries):
            n = 5
            countries[idx]
            countries[i]
            queues_in[process_n].put((idx, i, n))
            n_runs += 1

    for process in processess:
        process.start()

    data_points = 0
    last_entry_time = time.time()
    st = time.time()
    time.sleep(2)
    while True:
        if (data_points/n_runs) > 0.95 and (time.time() - last_entry_time) > 30 or data_points == n_runs:
            break
        for q in queues_out:
            while not q.empty():
                q_v = q.get()
                if not q_v:
                    break
                x, y, similarity, n = q_v
                similarities[x][y] = similarity
                similarities[y][x] = similarity
                n_series[y][x] = n
                n_series[x][y] = n
                data_points += 1
                last_entry_time = time.time()
                print(f"\r {data_points}/{n_runs} ETA: {(n_runs-data_points)*((time.time()-st) / data_points):.0f}s",
                      end='')

    for process in processess:
        process.kill()


    to_save = np.concatenate((countries_string,similarities))
    np.savetxt("data/similarity_measurements/country_similarities_1_1.csv", to_save, delimiter=";", fmt='%s', encoding='utf-8')

    return to_save

def find_country_similarity_all_series(kg, similarities):
    """
    This function will get the similarity between the top 5 similar countries to A, and calculate the similarity between
    them for all series.
    """

    countries = [s for s, p, o in kg.triples((None, None, RDF.country))]
    queues_in = [mp.Queue() for _ in range(CPU_N)]
    queues_out = [mp.Queue() for _ in range(CPU_N)]
    processess = [mp.Process(target=get_similarity_parallel, args=(kg, countries, queues_in[i], queues_out[i])) for i in
                  range(CPU_N)]
    top_n = 5
    n_runs = 0
    top_similar = np.zeros(shape=(len(countries), len(countries)), dtype=float)
    for x, row in enumerate(similarities[1:,:]):
        indices = np.argpartition(row, -top_n)[-top_n:]
        process_n = x % CPU_N
        for y in indices:
            queues_in[process_n].put((x, y, 0))
            n_runs += 1

    for process in processess:
        process.start()

    data_points = 0
    last_entry_time = time.time()
    st = time.time()
    while True:
        #Allow time out when at least 95% of the data has been acquired or if alls has been acquired
        if (data_points/n_runs) > 0.95 and (time.time() - last_entry_time) > 30 or data_points == n_runs:
            break
        for q in queues_out:
            while not q.empty():
                q_v = q.get()
                if not q_v:
                    break
                x, y, similarity, n = q_v
                top_similar[x][y] = similarity
                top_similar[y][x] = similarity

                data_points += 1
                last_entry_time = time.time()
                print(f"\r {data_points}/{n_runs} ETA: {(n_runs-data_points)*((time.time()-st) / data_points):.0f}s",
                      end='')

    for process in processess:
        process.kill()

    to_save = np.concatenate((np.array([country.split('/')[-1] for country in countries], ndmin=2),top_similar))
    np.savetxt("data/similarity_measurements/country_similarities_top_1_1.csv", to_save, delimiter=";", fmt='%s', encoding='utf-8')

    return to_save






def main():
    graph_path = 'data/graphs/kg.ttl'

    kg = Graph()
    try:
        print("Loading graph!", end='')
        kg.parse(graph_path)
        print("\rGraph loaded!")
    except FileNotFoundError:
        sys.exit("\rRDF not found, terminating program!")


    find_series_correlations(kg, ("Business", "Health"))
    sim = find_country_similarity_parallel(kg)
    find_country_similarity_all_series(kg, similarities=sim)
    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
