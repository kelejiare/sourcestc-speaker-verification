#!/usr/bin/env python3
"""Materialize exact plus vectors on frozen relation waveforms; CPU parallel to GPU KD."""
import argparse
import json
import sys
import tempfile
import time
import wave
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.relation_distillation import RECORD,read_cache,write_json
from audio_features import load_audio
from build_plus_extractor import (
    required,DurationAdaptiveEnsembleExtractor,assert_online_acoustic_contract,
    JointKDReweightExtractor,SourceAdaptedExtractor,DurationMetricExtractor,
    PlusFeatureFusionExtractor,
)


def build_anchor(device='cpu'):
    pt=required('GAME823_PT_MODEL_PATH',('model_pt.pt',))
    camp=required('GAME823_CAMPPLUS_MODEL_PATH',('campplus_cn_en_common.pt',))
    lm=required('GAME823_REDIM_LM_MODEL_PATH',('redimnet2-b6-vb2-vox2-cnc2-lm.pt',))
    ptn=required('GAME823_REDIM_PTN_MODEL_PATH',('redimnet2-b6-vb2-vox2-cnc2-ptn.pt',))
    adapter=required('GAME823_JOINT_ADAPTER_PATH',('joint_kd_adapter.npz',))
    mobile=required('GAME823_MOBILE_ADAPTER_PATH',('mobile_kd_adapter.npz',))
    short_metric=required('GAME823_SHORT_METRIC_ADAPTER_PATH',('short_duration_metric_adapter.npz',))
    short_pt=required('GAME823_SHORT_PT_ADAPTER_PATH',('short_duration_pt_adapter.npz',))
    plus=required('GAME823_REDIMNET2PLUS_MODEL_PATH',('redimnet2plus-b6-lmft-robust.pt',))
    base=DurationAdaptiveEnsembleExtractor(pt,camp,device_name=device)
    assert_online_acoustic_contract(base)
    joint=JointKDReweightExtractor(pt,camp,lm,ptn,adapter,device_name=device,base_extractor=base)
    return PlusFeatureFusionExtractor(DurationMetricExtractor(SourceAdaptedExtractor(joint,mobile),short_metric,short_pt),plus)


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev',choices=('dev','test'));p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(2)
    rows,payload=read_cache();indices=[i for i,r in enumerate(rows) if r['split']==a.split]
    rows=[rows[i] for i in indices];waves=payload['waves'][indices]
    out=RECORD/'evaluations'/a.split/'plus_online_anchor'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'embeddings.npz').exists():raise FileExistsError('anchor cache exists')
    extractor=build_anchor(a.device);values={k:[] for k in ('long3','short05','short10')}
    started=time.time();max_error=0.;audits=[]
    with tempfile.TemporaryDirectory(prefix='game823-relation-anchor-') as tmp:
        for index,(row,x) in enumerate(zip(rows,waves)):
            route='data_scenery2' if row['domain']=='meeting' else 'data_scenery1'
            path=Path(tmp)/route/'audio'/(row['id']+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
            for duration,n in [('long3',48000),('short05',8000),('short10',16000)]:
                y=x[(48000-n)//2:(48000+n)//2]
                pcm=np.clip(np.round(y.astype(np.float64)*2147483648),-2147483648,2147483647).astype('<i4')
                with wave.open(str(path),'wb') as wav:
                    wav.setnchannels(1);wav.setsampwidth(4);wav.setframerate(16000);wav.writeframes(pcm.tobytes())
                error=float(np.max(np.abs(load_audio(path,16000,False)-y)))
                max_error=max(max_error,error)
                assert error<1e-8,'cache waveform mismatch'
                vector,metadata=extractor.extract_file(path)
                assert vector.shape==(3520,) and np.isfinite(vector).all()
                assert str(metadata.get('fallback_mode','none')) in ('','none'), metadata
                assert metadata.get('auxiliary_mode')!='file_safe_bypass', metadata
                values[duration].append(vector)
                audits.append(dict(id=row['id'],duration=duration,route=route,
                    source_policy=metadata.get('source_policy'),duration_policy=metadata.get('duration_policy')))
            if index%10==0:
                print(json.dumps(dict(phase='plus_anchor',split=a.split,files=index+1,total=len(rows),seconds=time.time()-started)),flush=True)
    np.savez(out/'embeddings.npz',ids=np.asarray([r['id'] for r in rows]),**{k:np.stack(v) for k,v in values.items()})
    write_json(out/'RECEIPT.json',dict(seconds=time.time()-started,wave_roundtrip_max_abs=max_error,
        torch=torch.__version__,device=a.device,files=len(rows),audits=audits,hidden_score_guarantee=False))


if __name__=='__main__':main()
