"""Load only registered, eligible Ausgrid windows under qualified mirror provenance."""
import csv
from datetime import date,datetime,timedelta,timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile
from interval_meter import MeterReading
from screen_ausgrid import parse_date,interval_utc,non24_days,check_row


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(repository, customer, test_start):
    root=Path(repository);source=root/'data'
    adjudication=root/'data/metadata/source_adjudication.json'
    decision=json.loads(adjudication.read_text())
    if decision['decision']!='accept qualified retrospective preservation copy':
        raise ValueError('source not adjudicated for retrospective use')
    cohort_path=root/'data/metadata/external_cohort.json'
    if sha(cohort_path)!=decision['cohort_sha256']:raise ValueError('changed frozen cohort')
    cohort=json.loads(cohort_path.read_text())
    matches=[c for c in cohort['cases'] if c['customer']==customer and c['test_start']==test_start]
    if len(matches)!=1:raise ValueError('case not in registered eligible cohort')
    case=matches[0];start=date.fromisoformat(case['history_start']);end=date.fromisoformat(case['test_end_exclusive'])
    days=[start+timedelta(days=i) for i in range((end-start).days)]
    if non24_days(days):raise ValueError('unresolved transition day')
    archive_path=source/'raw/ausgrid.zip'
    if sha(archive_path)!=decision['source_archive_sha256']:raise ValueError('changed source archive')
    records={}
    with zipfile.ZipFile(archive_path) as archive:
        for member,digest in decision['member_sha256'].items():
            with archive.open(member) as f:
                h=hashlib.sha256()
                while chunk:=f.read(1024*1024):h.update(chunk)
            if h.hexdigest()!=digest:raise ValueError('changed source member')
            with archive.open(member) as stream:
                reader=csv.reader(io.TextIOWrapper(stream,encoding='utf-8-sig',newline=''))
                for _ in range(5):
                    header=next(reader)
                    if header and header[0].strip()=='Customer':break
                else:raise ValueError('missing CSV header')
                index={key.strip().lower():i for i,key in enumerate(header)}
                time_cols=[index[f'{((i+1)*30%1440)//60}:{((i+1)*30%1440)%60:02d}'] for i in range(48)]
                for row in reader:
                    if not row or not any(row):continue
                    if len(row)!=len(header):raise ValueError('CSV schema drift')
                    if int(row[index['customer']])!=customer:continue
                    day=parse_date(row[index['date']])
                    if not start<=day<end:continue
                    channel=row[index['consumption category']].strip()
                    if channel not in ('GC','GG','CL'):raise ValueError('unexpected channel')
                    key=(day,channel)
                    if key in records:raise ValueError('duplicate required row')
                    vals=[row[i] for i in time_cols]
                    if check_row(vals,row[index['row quality']])['reasons']:raise ValueError('invalid required row')
                    records[key]=[float(v) for v in vals]
    anchor=interval_utc(date.fromisoformat(test_start),0)[0]
    result=[]
    for day in days:
        if any((day,c) not in records for c in ('GC','GG','CL')):raise ValueError('missing measured channel')
        for i in range(48):
            a,b=interval_utc(day,i);t0=(a-anchor).total_seconds();t1=(b-anchor).total_seconds()
            result.append(MeterReading(t0,t1,records[(day,'GC')][i]+records[(day,'CL')][i],records[(day,'GG')][i],t1))
    return result,dict(customer=customer,test_start=test_start,anchor_utc=anchor.isoformat(),
                       record_count=len(result),source_archive_sha256=decision['source_archive_sha256'],
                       provenance='third-party preservation copy; not publisher-authenticated bytes',
                       ready_at='assumed at interval end for retrospective model; not historical receipt')
