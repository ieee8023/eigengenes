import glob
import numpy
import os
import pandas
import sys
import theano
import urllib
import zipfile

def load(organism, shared = True):
    """
    Downloads gene expression data for the specified ogranism from 
    Colombos website (http://www.colombos.net/) and prepares the 
    data for subsequent analysis. 

    Returns:
      expressions - M by N matrix, where M is the number of genes and N is
                    the number of contrasts. The matrix is/isn't Theano shared
                    if shared is True/False.
      genes       - a list of M gene names.
      contrasts   - a list of N contrast identifiers.
      refannot    - a dictionary mapping contrast identifiers to a set of 
                    reference conditions.
      testannot   - a dictionary mapping contrast identifiers to a set of 
                    test conditions.
    """

    source = "http://www.colombos.net/cws_data/compendium_data"
    zipfname = "%s_compendium_data.zip" %(organism)

    # Download data if necessary.
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.isfile("data/%s" %(zipfname)):
        print("Downloading %s data..." %(organism))
        urllib.urlretrieve("%s/%s" %(source, zipfname), "data/%s" %(zipfname))

    # Extract data.
    fh = zipfile.ZipFile("data/%s" %(zipfname))
    fh.extractall("data")
    fh.close()

    # Prepare data for later processing.
    print("Preparing %s data..." %(organism))
    expfname = glob.glob("data/colombos_%s_exprdata_*.txt" %(organism))[0]
    refannotfname = glob.glob("data/colombos_%s_refannot_*.txt" %(organism))[0]
    testannotfname = glob.glob("data/colombos_%s_testannot_*.txt" 
                               %(organism))[0]
    df = pandas.read_table(expfname, skiprows = 5, header = 1)
    df = df.fillna(0.0)
    genes = df["Gene name"].values
    expressions = df.iloc[:, 3:len(df.columns)].values
    contrasts = numpy.array(open(expfname, 
                                 "r").readline().strip().split('\t')[1:], 
                            dtype = object)
    lines = open(refannotfname, "r").readlines()
    refannot = {}
    for line in lines[1:]:
        contrast, annot = line.strip().split("\t")
        refannot.setdefault(contrast, set())
        refannot[contrast].add(annot)
    lines = open(testannotfname, "r").readlines()
    testannot = {}
    for line in lines[1:]:
        contrast, annot = line.strip().split("\t")
        testannot.setdefault(contrast, set())
        testannot[contrast].add(annot)

    # Remove extracted files.
    os.remove(expfname)
    os.remove(refannotfname)
    os.remove(testannotfname)

    if shared:
        return theano.shared(expressions, borrow = True), \
            genes, contrasts, refannot, testannot
    return expressions, genes, contrasts, refannot, testannot

def ecoli(shared):
    """ Escherichia coli (4321 genes and 4077 contrasts). """

    return load("ecoli", shared)

def bsubt(shared):
    """ Bacillus subtilis (4176 genes and 1259 contrasts). """

    return load("bsubt", shared)

def scoel(shared):
    """ Streptomyces coelicolor (8239 genes and 371 contrasts). """
    
    return load("scoel", shared)

def paeru(shared):
    """ Pseudomonas aeruginosa (5647 genes and 559 contrasts). """

    return load("paeru", shared)

def mtube(shared):
    """ Mycobacterium tuberculosis (4068 genes and 1098 contrasts). """

    return load("mtube", shared)

def hpylo(shared):
    """ Helicobacter pylori (1616 genes and 133 contrasts). """

    return load("hpylo", shared)

def meta_sente(shared):
    """ Salmonella enterica (cross-strain) (6261 genes and 1066 contrasts). """

    return load("meta_sente", shared)

def sente_lt2(shared):
    """ Salmonella enterica serovar Typhimurium LT2 (4556 genes and 172 
    contrasts). """

    return load("sente_lt2", shared)

def sente_14028s(shared):
    """ Salmonella enterica serovar Typhimurium 14028S (5416 genes and 681 
    contrasts). """

    return load("sente_14028s", shared)

def sente_sl1344(shared):
    """ Salmonella enterica serovar Typhimurium SL1344 (4655 genes and 213 
    contrasts). """

    return load("sente_sl1344", shared)

def smeli_1021(shared):
    """ Sinorhizobium meliloti (6218 genes and 424 contrasts). """

    return load("smeli_1021", shared)

def cacet(shared):
    """ Clostridium acetobutylicum (3778 genes and 377 contrasts). """

    return load("cacet", shared)

def tther(shared):
    """ Thermus thermophilus (2173 genes and 444 contrasts). """

    return load("tther", shared)

def banth(shared):
    """ Bacillus anthracis (5039 genes and 66 contrasts). """

    return load("banth", shared)

def bcere(shared):
    """ Bacillus cereus (5231 genes and 283 contrasts). """

    return load("bcere", shared)

def bthet(shared):
    """ Bacteroides thetaiotaomicron (4816 genes and 333 contrasts). """

    return load("bthet", shared)

def cjeju(shared):
    """ Campylobacter jejuni (1572 genes and 152 contrasts). """

    return load("cjeju", shared)

def lrham(shared):
    """ Lactobacillus rhamnosus (2834 genes and 79 contrasts). """

    return load("lrham", shared)

def mmari(shared):
    """ Methanococcus maripaludis (1722 genes and 364 contrasts). """

    return load("mmari", shared)

def sflex(shared):
    """ Shigella flexneri (4315 genes and 35 contrasts). """

    return load("sflex", shared)

def spneu(shared):
    """ Streptococcus pneumoniae (1914 genes and 68 contrasts). """

    return load("spneu", shared)

def ypest(shared):
    """ Yersinia pestis (3979 genes and 36 contrasts). """

    return load("ypest", shared)

def meta_ally2(shared):
    """ Cross-species analysis (31982 genes and 11224 contrasts). """

    return load("meta_ally2", shared)
