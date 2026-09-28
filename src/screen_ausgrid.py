"""Frozen quality-only screen; no controllers, forecasts, costs or window search."""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import date,datetime,timedelta,timezone
import hashlib
import io
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo
import zipfile


def parse_date(value):
    for fmt in ('%d/%m/%Y','%d%b%Y','%d-%b-%y','%d-%b-%Y','%Y-%m-%d'):
        try:return datetime.strptime(value.strip(),fmt).date()
        except ValueError:pass
    raise ValueError('unrecognized day-first source date: '+value)


def calendar_days(start,count):return [start+timedelta(days=i) for i in range(count)]


def non24_days(days):
    zone=ZoneInfo('Australia/Sydney');result=[]
    for day in days:
        begin=datetime.combine(day,datetime.min.time(),zone).astimezone(timezone.utc)
        end=datetime.combine(day+timedelta(days=1),datetime.min.time(),zone).astimezone(timezone.utc)
        hours=(end-begin).total_seconds()/3600
        if hours!=24:result.append({'date':day.isoformat(),'hours':hours})
    return result


def interval_utc(day,index):
    """48 end-labelled half hours; transition days are never forced into this grid."""
    if not 0<=index<48 or non24_days([day]):raise ValueError('invalid or unresolved DST interval')
    zone=ZoneInfo('Australia/Sydney')
    local=datetime.combine(day,datetime.min.time())
    start=(local+timedelta(minutes=index*30)).replace(tzinfo=zone).astimezone(timezone.utc)
    end=(local+timedelta(minutes=(index+1)*30)).replace(tzinfo=zone).astimezone(timezone.utc)
    if end-start!=timedelta(minutes=30):raise ValueError('non-half-hour interval')
    return start,end


def check_row(values,flag):
    reasons=[]
    if flag.strip():reasons.append('nonactual-or-unknown-quality')
    if len(values)!=48:reasons.append('not-48-readings')
    bad=0
    for value in values:
        try:
            x=float(value)
            if not math.isfinite(x) or x<0:bad+=1
        except (ValueError,TypeError):bad+=1
    if bad:reasons.append('invalid-energy')
    return {'reasons':reasons,'invalid_values':bad,'row_quality':flag.strip()}


def case_quality(records,customer,days,channels):
    missing=invalid=0
    reasons=Counter()
    for day in days:
        for channel in channels:
            r=records.get((customer,channel,day))
            if r is None:missing+=1;reasons['missing-'+channel]+=1
            elif r['reasons']:
                invalid+=1;reasons.update(r['reasons'])
    return {'complete_actual_nonnegative':not missing and not invalid,'missing_channel_days':missing,
            'invalid_channel_days':invalid,'reasons':dict(reasons)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=Path(__file__).parent/'ausgrid-screen-v1')
    args=parser.parse_args();base=args.package
    protocol_path=base/'screen_protocol.json';protocol=json.loads(protocol_path.read_text())
    preflight=json.loads((base/'screen_preflight.json').read_text())
    sha=lambda data:hashlib.sha256(data).hexdigest()
    if sha(Path(__file__).read_bytes())!=preflight['screen_source_sha256'] or sha(protocol_path.read_bytes())!=preflight['protocol_sha256']:
        raise ValueError('screen code/protocol changed after preflight')
    path=base/'source_docs/Ausgrid_mirror_archive.zip'
    if sha(path.read_bytes())!=protocol['source_archive_sha256']:raise ValueError('source archive changed')
    starts=[date.fromisoformat(s) for s in protocol['test_starts']]
    windows={s:calendar_days(s-timedelta(days=protocol['history_days']),protocol['history_days']+protocol['test_days']) for s in starts}
    inspected_days=set(d for days in windows.values() for d in days)
    records={};seen=set();metadata=defaultdict(set);row_counts=Counter();flag_counts=Counter();inspected=Counter();member_hashes={};headers={}
    with zipfile.ZipFile(path) as archive:
        for name in ['Solar home 2011-2012.csv','Solar home 2012-2013.csv']:
            h=hashlib.sha256()
            with archive.open(name) as stream:
                while chunk:=stream.read(1024*1024):h.update(chunk)
            member_hashes[name]=h.hexdigest()
            with archive.open(name) as stream:
                reader=csv.reader(io.TextIOWrapper(stream,encoding='utf-8-sig',newline=''))
                for _ in range(5):
                    header=next(reader)
                    if header and header[0].strip()=='Customer':break
                else:raise ValueError('source header absent')
                headers[name]=header
                index={key.strip().lower():i for i,key in enumerate(header)}
                for key in ['customer','generator capacity','postcode','consumption category','date','row quality']:
                    if key not in index:raise ValueError('required source column missing: '+key)
                time_columns=[]
                for i in range(48):
                    minutes=(i+1)*30%1440;label=f'{minutes//60}:{minutes%60:02d}'
                    if label not in index:raise ValueError('half-hour source column missing: '+label)
                    time_columns.append(index[label])
                for row in reader:
                    if not row or not any(row):continue
                    if len(row)!=len(header):raise ValueError('schema width drift')
                    customer=int(row[index['customer']]);channel=row[index['consumption category']].strip()
                    day=parse_date(row[index['date']]);key=(customer,channel,day)
                    if customer not in protocol['customer_universe'] or channel not in ('GC','GG','CL'):
                        raise ValueError('unexpected source customer/channel')
                    duplicate=key in seen;seen.add(key)
                    metadata[customer].add(channel);row_counts[channel]+=1
                    if day not in inspected_days:continue
                    checked=check_row([row[j] for j in time_columns],row[index['row quality']])
                    if duplicate:checked['reasons'].append('duplicate-customer-channel-day')
                    if key in records:checked['reasons']=sorted(set(checked['reasons']+records[key]['reasons']))
                    records[key]=checked;inspected[channel]+=1;flag_counts[checked['row_quality']]+=1
    cases=[]
    for start,days in windows.items():
        clock=non24_days(days)
        for customer in protocol['customer_universe']:
            whole=case_quality(records,customer,days,protocol['required_channels_whole_site'])
            general=case_quality(records,customer,days,protocol['required_channels_general_load_scope'])
            cases.append({'customer':customer,'test_start':start.isoformat(),'history_start':days[0].isoformat(),
                          'test_end_exclusive':(start+timedelta(days=protocol['test_days'])).isoformat(),
                          'whole_site_quality':whole,'general_load_only_quality':general,
                          'unresolved_dst_days':clock,'whole_site_clock_and_quality_candidate':whole['complete_actual_nonnegative'] and not clock,
                          'general_load_clock_and_quality_candidate':general['complete_actual_nonnegative'] and not clock,
                          'execution_eligible':False})
    aggregates=[]
    for start in protocol['test_starts']:
        subset=[x for x in cases if x['test_start']==start]
        aggregates.append({'test_start':start,'screened_customers':len(subset),
                           'whole_site_quality':sum(x['whole_site_quality']['complete_actual_nonnegative'] for x in subset),
                           'general_load_only_quality':sum(x['general_load_only_quality']['complete_actual_nonnegative'] for x in subset),
                           'whole_site_clock_and_quality_candidates':sum(x['whole_site_clock_and_quality_candidate'] for x in subset),
                           'general_load_clock_and_quality_candidates':sum(x['general_load_clock_and_quality_candidate'] for x in subset),
                           'unresolved_dst_days':subset[0]['unresolved_dst_days']})
    from datetime import datetime,timezone
    result={'screened_at_utc':datetime.now(timezone.utc).isoformat(),'controllers_run':False,
            'preflight':preflight,'member_sha256':member_hashes,'headers':headers,
            'source_rows_read_metadata_only_or_quality':dict(row_counts),'numeric_quality_rows_in_fixed_windows':dict(inspected),
            'row_quality_counts_in_fixed_windows':dict(flag_counts),'customers_with_observed_CL':sum('CL' in cs for cs in metadata.values()),
            'missing_CL_is_not_zero':True,'summaries':aggregates,'cases':cases,
            'distinct_whole_site_candidates':sorted({x['customer'] for x in cases if x['whole_site_clock_and_quality_candidate']}),
            'clock_note':'48 civil-time slots do not resolve 23/25-hour transition days; no windows shifted',
            'source_note':'Third-party archive with matching notes from two copies; no original publisher CSV checksum available',
            'remaining_year_energy_values':'not numerically analyzed; not used for tuning',
            'activation':False}
    with (base/'quality_screen.json').open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    with (base/'window_summary.csv').open('x',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(aggregates[0]));writer.writeheader();writer.writerows(aggregates)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','headers','preflight','distinct_whole_site_candidates')},indent=2))
    print('Distinct whole-site quality/clock candidates:',len(result['distinct_whole_site_candidates']))


if __name__=='__main__':main()
