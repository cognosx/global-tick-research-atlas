# ============================================================
#  TARGETED RE-ENRICHMENT: affiliations + countries (Colab)
#  Runs ONLY on canonical studies missing country/institution.
#  Sources: fresh OpenAlex retry, then Crossref affiliation strings
#  -> parse country via a country-name/adjective lookup.
#  Realistic: fixes records that have data SOMEWHERE; cannot
#  invent affiliations OpenAlex/Crossref simply don't hold.
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
!pip -q install requests pandas openpyxl

import os, re, time
import pandas as pd, numpy as np, requests

FOLDER  = "/content/drive/MyDrive/ticks"  # <-- EDIT
OUT_DIR = os.path.join(FOLDER, "preprocessing_data")
IN_PATH = os.path.join(OUT_DIR, "master_canonical_clean.xlsx")   # use cleaned file if you ran the type audit
if not os.path.exists(IN_PATH): IN_PATH = os.path.join(OUT_DIR,"master_canonical.xlsx")
SHEET = "canonical"
POLITE_EMAIL = "cognosxx@gmail.com"
CHECKPOINT_EVERY = 400
CKPT = os.path.join(OUT_DIR, "reenrich_aff_checkpoint.parquet")

df = pd.read_excel(IN_PATH, sheet_name=SHEET, dtype=str).fillna("")
print("Canonical studies:", len(df))

for c in ["oa_countries","oa_institutions","affiliations","doi","oaid_norm"]:
    if c not in df.columns: df[c]=""

# resume
done=set()
if os.path.exists(CKPT):
    prev=pd.read_parquet(CKPT).set_index("record_id")
    df=df.set_index("record_id"); df.update(prev); df=df.reset_index()
    done=set(prev.index[prev.get("aff_reenriched","").eq("1")].tolist())
    print("Resumed:", len(done))
if "aff_reenriched" not in df.columns: df["aff_reenriched"]=""

# --- country parser from free-text affiliation strings (explicit, no pycountry dependency) ---
COUNTRIES = {
 "nigeria":"NG","kenya":"KE","ghana":"GH","tanzania":"TZ","ethiopia":"ET","uganda":"UG",
 "south africa":"ZA","cameroon":"CM","senegal":"SN","ivory coast":"CI","cote d ivoire":"CI",
 "mali":"ML","burkina faso":"BF","zambia":"ZM","zimbabwe":"ZW","mozambique":"MZ","angola":"AO",
 "rwanda":"RW","benin":"BJ","botswana":"BW","namibia":"NA","malawi":"MW","sudan":"SD",
 "democratic republic of congo":"CD","drc":"CD","gabon":"GA","chad":"TD","niger":"NE",
 "somalia":"SO","madagascar":"MG","sierra leone":"SL","togo":"TG","guinea":"GN","eritrea":"ER",
 "burundi":"BI","lesotho":"LS","eswatini":"SZ","gambia":"GM","liberia":"LR","mauritania":"MR",
 "germany":"DE","france":"FR","united kingdom":"GB","uk":"GB","england":"GB","scotland":"GB",
 "wales":"GB","spain":"ES","italy":"IT","netherlands":"NL","belgium":"BE","switzerland":"CH",
 "sweden":"SE","norway":"NO","denmark":"DK","finland":"FI","poland":"PL","czech republic":"CZ",
 "czechia":"CZ","austria":"AT","portugal":"PT","greece":"GR","hungary":"HU","romania":"RO",
 "slovakia":"SK","slovenia":"SI","croatia":"HR","serbia":"RS","bulgaria":"BG","ireland":"IE",
 "russia":"RU","ukraine":"UA","united states":"US","usa":"US","canada":"CA","mexico":"MX",
 "brazil":"BR","argentina":"AR","colombia":"CO","chile":"CL","peru":"PE","venezuela":"VE",
 "ecuador":"EC","bolivia":"BO","uruguay":"UY","china":"CN","japan":"JP","south korea":"KR",
 "india":"IN","pakistan":"PK","bangladesh":"BD","iran":"IR","iraq":"IQ","turkey":"TR",
 "israel":"IL","australia":"AU","new zealand":"NZ","egypt":"EG","morocco":"MA","tunisia":"TN",
 "algeria":"DZ","libya":"LY","saudi arabia":"SA","thailand":"TH","vietnam":"VN","malaysia":"MY",
 "indonesia":"ID","philippines":"PH","sri lanka":"LK","nepal":"NP","kazakhstan":"KZ",
}
def countries_from_text(txt):
    t=re.sub(r"[^a-z ]"," "," "+str(txt or "").lower()+" "); t=" "+re.sub(r"\s+"," ",t)+" "
    found=[]
    for name,code in COUNTRIES.items():
        if (" "+name+" ") in t and code not in found: found.append(code)
    return found

sess=requests.Session(); sess.headers.update({"User-Agent":f"reenrich/1.0 (mailto:{POLITE_EMAIL})"})
def get(url):
    try:
        r=sess.get(url,timeout=30)
        if r.status_code==200: return r.json()
    except Exception: return None
    return None
def norm_doi(d):
    d=str(d or "").strip().lower()
    for p in ("https://doi.org/","doi:"): d=d.replace(p,"")
    return d if d.startswith("10.") else ""

# target: canonical rows missing countries OR institutions, that have a DOI to look up
need = df[(df["oa_countries"].eq("")|df["oa_institutions"].eq("")) &
          (df["doi"].map(norm_doi).ne("")) & (~df["record_id"].isin(done))]
print(f"Records to attempt: {len(need)} (missing country/institution, have DOI)")

filled_oa=filled_cr=filled_country=0
for n,(idx,row) in enumerate(need.iterrows()):
    doi=norm_doi(row["doi"])
    # 1) OpenAlex retry (sometimes affiliations populated since first run, or first run missed)
    if row["oa_institutions"]=="":
        w=get(f"https://api.openalex.org/works/https://doi.org/{doi}?mailto={POLITE_EMAIL}")
        time.sleep(0.12)
        if w and not w.get("error"):
            insts=[]; ctry=[]
            for a in (w.get("authorships") or []):
                for ins in (a.get("institutions") or []):
                    if ins.get("display_name"): insts.append(ins["display_name"])
                    if ins.get("country_code"): ctry.append(ins["country_code"])
            if insts:
                df.at[idx,"oa_institutions"]="; ".join(dict.fromkeys(insts)); filled_oa+=1
            if ctry and df.at[idx,"oa_countries"]=="":
                df.at[idx,"oa_countries"]="; ".join(dict.fromkeys(ctry))
    # 2) Crossref affiliations (free-text) -> parse country
    if df.at[idx,"affiliations"]=="" or df.at[idx,"oa_countries"]=="":
        j=get(f"https://api.crossref.org/works/{doi}?mailto={POLITE_EMAIL}"); time.sleep(0.05)
        if j:
            affs=[]
            for a in j.get("message",{}).get("author",[]):
                for af in a.get("affiliation",[]):
                    if af.get("name"): affs.append(af["name"])
            if affs:
                if df.at[idx,"affiliations"]=="":
                    df.at[idx,"affiliations"]=" | ".join(dict.fromkeys(affs)); filled_cr+=1
                if df.at[idx,"oa_countries"]=="":
                    cc=[]
                    for a in affs: cc += countries_from_text(a)
                    if cc:
                        df.at[idx,"oa_countries"]="; ".join(dict.fromkeys(cc)); filled_country+=1
    df.at[idx,"aff_reenriched"]="1"
    if n and n % CHECKPOINT_EVERY==0:
        df.to_parquet(CKPT,index=False); print(f"  ...{n}/{len(need)} (oa+{filled_oa} cr+{filled_cr} ctry+{filled_country})")

df.to_parquet(CKPT,index=False)

# also parse countries from ANY existing affiliation text we now have but still lack country
m = df["oa_countries"].eq("") & df["affiliations"].ne("")
for idx in df.index[m]:
    cc=countries_from_text(df.at[idx,"affiliations"])
    if cc: df.at[idx,"oa_countries"]="; ".join(dict.fromkeys(cc))

def covpct(c): return round((df[c].astype(str).str.len()>0).mean()*100,1)
print("\n=== Coverage AFTER targeted re-enrichment (canonical set) ===")
for c in ["oa_countries","oa_institutions","affiliations"]:
    print(f"  {c:16s} {covpct(c)}%")
print(f"\nFilled: OpenAlex institutions +{filled_oa} | Crossref affiliations +{filled_cr} | countries parsed +{filled_country}")

def safe_xlsx(frame,path,sheet):
    ILLEGAL=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out=frame.copy()
    for c in out.columns: out[c]=out[c].map(lambda v:ILLEGAL.sub(" ",str(v))[:32000])
    with pd.ExcelWriter(path,engine="openpyxl") as xw: out.to_excel(xw,sheet_name=sheet,index=False)
OUTP=os.path.join(OUT_DIR,"master_canonical_enriched.xlsx")
safe_xlsx(df.drop(columns=[c for c in ["aff_reenriched"] if c in df.columns]), OUTP, "canonical")
print(f"\nSaved {OUTP} -> use THIS for the bibliometric analysis")
