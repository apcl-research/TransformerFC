import sys
import pickle
import argparse
import re
import os

import collections

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from stemming.lovins import stem as lovins_stemmer

from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def bleu_so_far(refs, preds):
    Ba = corpus_bleu(refs, preds)
    B1 = corpus_bleu(refs, preds, weights=(1,0,0,0))
    B2 = corpus_bleu(refs, preds, weights=(0,1,0,0))
    B3 = corpus_bleu(refs, preds, weights=(0,0,1,0))
    B4 = corpus_bleu(refs, preds, weights=(0,0,0,1))

    Ba = round(Ba * 100, 2)
    B1 = round(B1 * 100, 2)
    B2 = round(B2 * 100, 2)
    B3 = round(B3 * 100, 2)
    B4 = round(B4 * 100, 2)

    ret = ''
    ret += ('for %s functions\n' % (len(preds)))
    ret += ('Ba %s\n' % (Ba))
    ret += ('B1 %s\n' % (B1))
    ret += ('B2 %s\n' % (B2))
    ret += ('B3 %s\n' % (B3))
    ret += ('B4 %s\n' % (B4))
    
    return ret

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str, default=None)
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/funcom/data/javastmt/output')  
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--challenge', action='store_true', default=False)
    parser.add_argument('--obfuscate', action='store_true', default=False)
    parser.add_argument('--sbt', action='store_true', default=False)
    parser.add_argument('--lim-overlap', dest='limoverlap', type=int, default=-1)
    parser.add_argument('--lim-overlap-sdats', dest='limoverlapsdats', type=int, default=-1)
    parser.add_argument('--tdats-filename', dest='tdatsfilename', type=str, default='tdats.test')
    parser.add_argument('--sdats-filename', dest='sdatsfilename', type=str, default='sdats.test')
    parser.add_argument('--coms-filename', dest='comsfilename', type=str, default='coms.test')
    parser.add_argument('--sentence-bleus', dest='sentencebleus', action='store_true', default=False)
    parser.add_argument('--delim', dest='delim', type=str, default='<SEP>')
    parser.add_argument('--max-preds', dest='maxpreds', type=int, default=10000000)

    args = parser.parse_args()
    outdir = args.outdir
    dataprep = args.dataprep
    input_file = args.input
    lim_overlap = args.limoverlap
    lim_overlap_sdats = args.limoverlapsdats
    tdatsfilename = args.tdatsfilename
    sdatsfilename = args.sdatsfilename
    comsfilename = args.comsfilename
    sentencebleus = args.sentencebleus
    delim = args.delim
    maxpreds = args.maxpreds

    if input_file is None:
        print('Please provide an input file to test')
        exit()

    if lim_overlap != -1 or lim_overlap_sdats != -1:
        prep('preparing tdats list... ')
        tdats = dict()
        tdatsf = open('%s/%s' % (dataprep, tdatsfilename), 'r')
        for c, line in enumerate(tdatsf):
            (fid, tdat) = line.split(delim)
            fid = int(fid)
            tdat = tdat.split()
            tdat = fil(tdat)
            tdats[fid] = tdat
        tdatsf.close()
        drop()
    
    if lim_overlap_sdats != -1:
        prep('preparing sdats list... ')
        sdats = dict()
        sdatsf = open('%s/%s' % (dataprep, sdatsfilename), 'r')
        for c, line in enumerate(sdatsf):
            (fid, sdat) = line.split(delim)
            fid = int(fid)
            sdat = sdat.split()
            sdat = fil(sdat)
            sdats[fid] = sdat
        sdatsf.close()
        drop()

    prep('preparing predictions list... ')
    preds = dict()
    predicts = open(input_file, 'r')
    for c, line in enumerate(predicts):
        try:
            (fid, pred) = line.split('\t')
            fid = int(fid)
        except:
            continue
        pred = pred.split()
        pred = fil(pred)
        preds[fid] = pred
        if c > maxpreds:
            break
    predicts.close()
    drop()

    re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

    if(sentencebleus):
        bfn = os.path.basename(input_file)
        bfn = os.path.splitext(bfn)[0]
        bleusf = open('{}/bleus/{}.tsv'.format(outdir, bfn), 'w')

    refs = list()
    newpreds = list()
    d = 0
    targets = open('%s/%s' % (dataprep, comsfilename), 'r')
    for line in targets:
        (fid, com) = line.split(delim)
        fid = int(fid)
        com = com.split()
        com = fil(com)
        
        if len(com) < 1:
            continue
            
        #if 'test' not in tdats[fid][:12]:
        #    continue
        
        #if com[0] != 'test' and com[1] != 'of':
        #    continue
        
        #print('fid ', fid)
        #print('com ', ' '.join(com))
        #print('pred', ' '.join(preds[fid]))
        #print('tdat', ' '.join(tdats[fid][:12]))
        #print()

        if lim_overlap_sdats > -1:
            #s = list(set(com) & set(sdats[fid]))
            #t = list(set(com) & set(tdats[fid]))
            #st = list(set(sdats[fid]) & set(tdats[fid]))
            #st = set(sdats[fid]).difference(tdats[fid]) # words in sdats, not tdats
            
            # remove the tdats from the sdats
            a_multiset = collections.Counter(sdats[fid][:100])
            b_multiset = collections.Counter(tdats[fid])
            #overlap = list((a_multiset & b_multiset).elements())
            a_remainder = list((a_multiset - b_multiset).elements())
            #b_remainder = list((b_multiset - a_multiset).elements())
            
            #st = set(a_remainder).difference(set(tdats[fid])) # words in sdats, not tdats
            
            #print(st)
            #print(a_remainder)
            
            s = set(set(com) & set(a_remainder)) # words in sdats and coms
            t = set(set(com) & set(tdats[fid]))
            o = set(set(a_remainder) - set(tdats[fid]))
            s_o = set(set(o) & set(com)) # words in sdats (but not tdats) and coms
            
            #print(s_o)
            
            #ssdats = ' '.join(sdats[fid])
            #stdats = ' '.join(tdats[fid])
            #sdats_without_tdats = ssdats.replace(stdats, '')
            #print(ssdats)
            #print(stdats)
            #print(sdats_without_tdats)
            
            #st = set(sdats[fid]).difference(set(tdats[fid][:12]))
            #print(sdats[fid])
            #print(tdats[fid])
            
            #print(s)
            #print(t)
            #quit()
            
            o_s = len(s_o)
            o_t = len(t)
            #o_st = len(st)
            
            #if not(o_s > 0 and o_st < lim_overlap_sdats):
            #if not(o_s == lim_overlap_sdats):
            if not(o_s > lim_overlap_sdats):
                continue
            else:
                print(s_o)

        if lim_overlap > -1:
            
            #try:
            #    com_s = [lovins_stemmer(w) for w in com]
            #    tdats_s = [lovins_stemmer(w) for w in tdats[fid][:12]]
            #except Exception as ex:
            #    continue
            
            t = list(set(com) & set(tdats[fid]))#[:12]))
            #t = list(set(com_s) & set(tdats_s))
            overlap = len(t) #/ len(set(com))
            
            #if overlap != lim_overlap:
            #if overlap < lim_overlap:
            if not(overlap < lim_overlap):
            #if overlap <4:
                continue

            #print('fid ', fid)
            #print('com ', ' '.join(com))
            #print('pred', ' '.join(preds[fid]))
            #print('tdat', ' '.join(tdats[fid][:12]))
            #print()

        try:
            newpreds.append(preds[fid])
            
            if(sentencebleus):
                
                Bas = corpus_bleu([[com]], [preds[fid]])
                B1s = corpus_bleu([[com]], [preds[fid]], weights=(1,0,0,0))
                B2s = corpus_bleu([[com]], [preds[fid]], weights=(0,1,0,0))
                B3s = corpus_bleu([[com]], [preds[fid]], weights=(0,0,1,0))
                B4s = corpus_bleu([[com]], [preds[fid]], weights=(0,0,0,1))

                Bas = round(Bas * 100, 4)
                B1s = round(B1s * 100, 4)
                B2s = round(B2s * 100, 4)
                B3s = round(B3s * 100, 4)
                B4s = round(B4s * 100, 4)
                
                bleusf.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(fid, Bas, B1s, B2s, B3s, B4s))
            
        except Exception as ex:
            #newpreds.append([])
            continue

        refs.append([com])
    
    if(sentencebleus):
        bleusf.close()

    print('final status')
    print(bleu_so_far(refs, newpreds))

