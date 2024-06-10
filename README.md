# BRCA Landscape in Egyptian women with cancer
# Introduction
The pathogenicity prediction of variant of uncertain significance (VUS) is challenging in understudied ancestries such as Northern African. The key challenges is lack of genetic epidemiology, case/control, segregation and phenotype data. We developed a reductionist approach relying on computational prediction and functional annotations.    
# Computational prediction of intronic variants using SpliceAI
To predict the functional impact on splicing, three methods can be applied:  
1) Run the tool for a custom set of variants 
Install SpliceAI from https://github.com/Illumina/SpliceAI and use the following script:

```
from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import numpy as np

input_sequence = 'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT'
# Replace this with your custom sequence

context = 10000
paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
models = [load_model(resource_filename('spliceai', x)) for x in paths]
x = one_hot_encode('N'*(context//2) + input_sequence + 'N'*(context//2))[None, :]
y = np.mean([models[m].predict(x) for m in range(5)], axis=0)

acceptor_prob = y[0, :, 1]
donor_prob = y[0, :, 2]
```

2) Use the precomputed scores: 
The annotations for all possible substitutions, 1 base insertions, and 1-4 base deletions within genes are available [here](https://basespace.illumina.com/s/otSPW8hnhaZR) for download 

3) For specific variants, can use the web-based online tool
https://spliceailookup.broadinstitute.org/ 
 
# Computational prediction of missense variants using MVP
MVP is deep learning based tool available at: 
https://github.com/ShenLab/missense
Precomputed scores are available at:  https://www.dropbox.com/s/d9we7gx42b7yatg/MVP_score_hg19.txt.bz2?dl=0.
Code to get MVP prediction for all missense variants: 
import csv
import os
import numpy as np
import pandas as pd
import pysam
from collections import defaultdict
from models import CNN_Model

Using TensorFlow backend.

this is a version 46 features in total, no RVIS no mis_badness !

split the file

# split the files into HS and HIS

pli = np.load('../data/gene/pli.npy').item()
HS_gene = set(gene for gene, pli_score in pli.iteritems() if pli_score < 0.5)
HIS_gene = set(gene for gene, pli_score in pli.iteritems() if pli_score >= 0.5)

fname = '/data/hq2130/large_files/rare_missense_id.anno.rare.All.reformat.csv'
f1 = '/data/hq2130/large_files/rare_missense_id.anno.rare.HIS.reformat.csv'
f2 = '/data/hq2130/large_files/rare_missense_id.anno.rare.HS.reformat.csv'
f3 = '/data/hq2130/large_files/rare_missense_id.anno.rare.nopli.reformat.csv'
with open(fname, 'rU') as f:
    r = csv.reader(f)
    head = r.next()

    fw1 = open(f1, 'wb')
    w1 = csv.writer(fw1)
    w1.writerow(head)

    fw2 = open(f2, 'wb')
    w2 = csv.writer(fw2)
    w2.writerow(head)

    fw3 = open(f3, 'wb')
    w3 = csv.writer(fw3)
    w3.writerow(head)

    for line in r:
        variant = dict(zip(head, line))
        if variant['genename'] in HIS_gene:
            w1.writerow(line)
        elif variant['genename'] in HS_gene:
            w2.writerow(line)
        else:
            w3.writerow(line)

    fw1.close()
    fw2.close()
    fw3.close()

# split the files
files = [f1, f2, f3]
for fname in files:
    idx, lines_per_file, count = 0, 1350000, 0
    with open(fname, 'rU') as f:
        r = csv.reader(f)
        head = r.next()
        for line in r:
            if count == 0:
                outname = fname.split('.csv')[0] + '_idx_' + str(idx) + '.csv'
                fw = open(outname, 'wb')
                w = csv.writer(fw)
                w.writerow(head)
                w.writerow(line)
                count += 1
            else:
                count += 1
                w.writerow(line)
                if count > lines_per_file:
                    fw.close()
                    idx += 1
                    count = 0
        fw.close()

add MVP annotation

# HIS prediction
prefix = 'HIS.'

# weight used for MVP model
weights_path = '../models/res_HIS_best_weight.hdf5'
exclude_cols = {'var_id', 'aaref', 'aaalt', 'target', 'Ensembl_transcriptid',
                'ref', 'alt', 'category',
                'source', 'INFO', 'disease', 'genename',
                '#chr', 'pos(1-based)',  'hg19_chr', 'hg19_pos(1-based)',
                'CADD_phred', '1000Gp3_AF', 'ExAC_AF', 'gnomad',
                'RVIS', 'mis_badness', 'MPC', 'REVEL', 'domino'}

dir_all = '/data/hq2130/large_files/'
fins = []
for fname in os.listdir(dir_all):
    if fname.startswith('rare_missense_id.anno.rare.HIS.reformat_idx') and 'cnn' not in fname:
        fins.append(dir_all + fname)


fouts = [f.split('.csv')[0] + '.cnn.csv' for f in fins]

for fin, fout in zip(fins, fouts):
    model = CNN_Model_Mode(weights_path=weights_path,
                           exclude_cols=exclude_cols,
                           train_flag=False,
                           verbose=0,
                           fname=fin,
                           f_out=fout)
    model.pred(get_last_layer=get_last_layer, layer_index=layer_index)

rank calculation

# generate per variant info file
dir_all = '/data/hq2130/large_files/'
bg_file = dir_all + 'All_rare_missense_best0208.txt'
with open(bg_file, 'w') as fw:
    head = ['CHROM', 'POS', 'REF', 'ALT', 'aaref', 'aaalt', 'genename',
            'Ensembl_transcriptid', 'pli', 'ExAC_AF', 'gnomad_exome',
            'CADD', 'REVEL', 'MPC', 'M-CAP_rankscore', 'MetaSVM_rankscore',
            'VEST3_rankscore', 'MVP_score']
    fw.write('\t'.join(head) + '\n')
    for fname in os.listdir(dir_all):
        # this concatenate both his and hs into the all rare missense
        if 'cnn' in fname and 'rare_missense_id' in fname:
            with open(dir_all + fname, 'rU') as f:
                r = csv.reader(f)
                head = r.next()
                for line in r:
                    variant = dict(zip(head, line))

                    # some variants with MPC of NA 12:104742193
                    if variant.get('MPC', '-1') == '':
                        variant['MPC'] = '-1'

                    info = [variant['hg19_chr'],
                            variant['hg19_pos(1-based)'],
                            variant['ref'],
                            variant['alt'],
                            variant['aaref'],
                            variant['aaalt'],
                            variant['genename'],
                            variant['Ensembl_transcriptid'],
                            variant['pli'],
                            variant['ExAC_AF'],
                            variant['gnomad_exome'],
                            variant.get('CADD_phred', '-1'),
                            variant.get('REVEL', '-1'),
                            variant.get('MPC', '-1'),
                            variant.get('M-CAP_rankscore', '-1'),
                            variant.get('MetaSVM_rankscore', '-1'),
                            variant.get('VEST3_rankscore', '-1'),
                            variant.get('cnn_prob', '-1')]
                    fw.write('\t'.join(info) + '\n')

rank in each method

pli = np.load('../data/gene/pli.npy').item()
HIS_gene = set(gene for gene, pli_score in pli.iteritems() if pli_score >= 0.5)

def count2rank(score2count):
    # # higher value means top rank, set missing value(-1) to be rank 1
    score2rank = {'CADD': {-1: 0.0},
                  'REVEL': {-1: 0.0},
                  'MPC': {-1: 0.0},
                  'M-CAP_rankscore': {-1: 0.0},
                  'MetaSVM_rankscore': {-1: 0.0},
                  'VEST3_rankscore': {-1: 0.0},
                  'MVP_score': {-1: 0.0}}
    for method in score2count:
        total = float(sum(score2count[method].values()))
        cur_total = 0
        scores = sorted(score2count[method].keys())
        for score in scores:
            cur_total += score2count[method][score]
            score2rank[method][score] = cur_total / total
    return score2rank

#canonical only?
#add rank here, choose 5 dight, based on counts to rank
with open('/data/hq2130/large_files/All_rare_missense_best0208.txt') as f:
    head = f.readline().strip().split()

    score2count_HIS = {'CADD': defaultdict(lambda: 0),
                       'REVEL': defaultdict(lambda: 0),
                       'MPC': defaultdict(lambda: 0),
                       'M-CAP_rankscore': defaultdict(lambda: 0),
                       'MetaSVM_rankscore': defaultdict(lambda: 0),
                       'VEST3_rankscore': defaultdict(lambda: 0),
                       'MVP_score': defaultdict(lambda: 0)}

    score2count_HS = {'CADD': defaultdict(lambda: 0),
                       'REVEL': defaultdict(lambda: 0),
                       'MPC': defaultdict(lambda: 0),
                       'M-CAP_rankscore': defaultdict(lambda: 0),
                       'MetaSVM_rankscore': defaultdict(lambda: 0),
                       'VEST3_rankscore': defaultdict(lambda: 0),
                       'MVP_score': defaultdict(lambda: 0)}

    methods = ['CADD', 'REVEL', 'MPC', 'M-CAP_rankscore','MetaSVM_rankscore',
               'VEST3_rankscore',  'MVP_score']
    for line in f:
        info = dict(zip(head, line.strip().split()))
        for method in methods:
            score = round(float(info[method]), 5)
            if score != -1:  # missing score not included in rank calc
                if info['genename'] in HIS_gene:
                    score2count_HIS[method][score] += 1
                else:
                    score2count_HS[method][score] += 1


score2rank_HIS = count2rank(score2count_HIS)
score2rank_HS = count2rank(score2count_HS)
np.save('/data/hq2130/large_files/score2rank_HIS_1pct', score2rank_HIS)
np.save('/data/hq2130/large_files/score2rank_HS_1pct', score2rank_HS)

with open('/data/hq2130/large_files/All_rare_missense_best0208.txt') as f, open('/data/hq2130/large_files/MVP_scores.txt', 'w') as fw:
    head = f.readline().strip().split()
    new_head = ['#CHROM', 'POS', 'REF', 'ALT', 'aaref', 'aaalt', 'GeneSymbol',
                'Ensembl_transcriptid', 'MVP_score', 'MVP_rank']
    fw.write('\t'.join(new_head) + '\n')

    for line in f:
        info = dict(zip(head, line.strip().split()))
        gene = info['genename']
        mvp_score = round(float(info['MVP_score']), 5)
        if gene in HIS_gene:
            mvp_rank = score2rank_HIS['MVP_score'][mvp_score]
        else:
            mvp_rank = score2rank_HS['MVP_score'][mvp_score]
        new_line = [info['CHROM'],
                    info['POS'],
                    info['REF'],
                    info['ALT'],
                    info['aaref'],
                    info['aaalt'],
                    info['genename'],
                    info['Ensembl_transcriptid'],
                    info['MVP_score'],
                    str(mvp_rank)]
        fw.write('\t'.join(new_line) + '\n')

%% bash
sort -k1,1V - k2,2n -T tmp  MVP_scores.txt > MVP_scores_sorted.txt
gzip /data/hq2130/large_files/MVP_scores_sorted.txt

