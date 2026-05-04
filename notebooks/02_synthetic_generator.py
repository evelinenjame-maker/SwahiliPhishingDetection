"""
Day 2 - Step 6: Synthetic Swahili phishing message generator.

Generates ~400 realistic Tanzanian phishing messages across 5 categories,
using observed BongoSCAM patterns combined with Mambina (2022) Swahili
smishing taxonomy.

Run from project root:
    python notebooks\02_synthetic_generator.py

Output:
    data/processed/synthetic_phishing.csv
"""

from pathlib import Path
import random
import itertools
import pandas as pd

random.seed(42)  # reproducibility - cite this in your dissertation methodology

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------
# Slot fillers - common Tanzanian context
# -----------------------------------------------------------------
SWAHILI_NAMES = [
    "ALLY ISSA", "OLIVA MATIAS", "PETER NYANGE", "JOHN MWALIMU",
    "MARY KIMARO", "JUMA HASSAN", "ASHA MOHAMED", "EMMANUEL JOSEPH",
    "RAJABU ABDULRAHMAN", "NEEMA SAMSON", "GODFREY MASSAWE",
    "ZAINABU SALEH", "ELIZABETH MWAKALINGA", "SAID BAKARI",
    "FATUMA OMARY", "BARAKA MAKWAIA", "DAVID KAGOMBA",
]

MOBILE_OPERATORS = ["Vodacom", "Airtel", "Tigo", "Halotel", "TTCL", "M-Pesa", "Tigopesa", "Airtel Money", "Halopesa"]

PHONE_NUMBERS = [
    f"0{prefix}{random.randint(1000000, 9999999)}"
    for prefix in ["65", "67", "68", "69", "75", "76", "78"]
    for _ in range(3)
]

AMOUNTS = ["50000", "100000", "150000", "200000", "250000", "300000",
           "500000", "750000", "1000000", "elfu hamsini", "laki moja",
           "laki mbili", "laki tatu", "milioni moja"]

URGENCY_WORDS = ["haraka", "sasa hivi", "kabla ya saa moja", "leo hii",
                 "muda mfupi", "ASAP", "haraka sana"]

# -----------------------------------------------------------------
# Category 1: Mobile money redirection (most common in BongoSCAM)
# -----------------------------------------------------------------
MOBILE_MONEY_TEMPLATES = [
    "Naomba unitumie {amount} kwenye namba hii ya {operator} {phone}. jina ({name})",
    "Iyo hela tuma humu kwenye {operator} {phone} jina lije {name}",
    "Mpenzi tuma {amount} {urgency} kwenye {operator} {phone} jina {name}",
    "Tafadhali tuma pesa kwenye {operator} {phone} jina litakuja {name} {urgency}",
    "Niko nje ya mtandao, tuma {amount} kwa namba hii ya {operator} {phone}, jina {name}",
    "Nimeshindwa kutoa pesa, tafadhali nitumie kwenye {operator} {phone} jina {name}",
    "Nikutumie pesa lakini namba yangu haifanyi kazi, tumia hii {phone} ({operator}) jina {name}",
    "Mke wangu yupo hospitali, naomba msaada wa {amount} kwenye {operator} {phone} jina {name}",
    "Au nitumie kwenye {operator} Namba.{phone} jina litakuja {name}",
    "Nipo safarini, tafadhali tuma {amount} {urgency} kwenye {operator} {phone} jina {name}",
]

# -----------------------------------------------------------------
# Category 2: Freemason / get-rich-quick recruitment
# -----------------------------------------------------------------
FREEMASON_TEMPLATES = [
    "666,KARIBU FREEMASON UTIMIZE NDOTO KATIKA BIASHARA, KILIMO,UFUGAJI,MICHEZO N.K KWA MHITAJI KUJIUNGA PG: {phone} AU {phone2}",
    "JOIN FREEMASON UPATE UTAJIRI WA HARAKA, BIASHARA, KAZI, UPENDO. PIGA SIMU {phone}",
    "MAFANIKIO YA HARAKA! Ungana na FREEMASON kwa biashara, ndoa, kazi. Tuma SMS kwa {phone}",
    "Karibu kwa MGANGA wa kienyeji. Tatua matatizo ya pesa, mapenzi, kazi. Piga {phone}",
    "DAWA YA MAFANIKIO IPO! Biashara, kilimo, kazi, mitihani. Tafadhali piga {phone} sasa",
    "MAJINI YA UTAJIRI YANAPATIKANA! Tafuta msaada haraka kwa {phone} - MGANGA MKUU",
    "Tarehe ya leo ni nzuri kujiunga FREEMASON. Faida: utajiri, ndoa, kazi. Wasiliana {phone}",
    "DAWA YA KUZUIA MAJANGA na kuvutia mafanikio. Piga simu sasa: {phone} au {phone2}",
]

# -----------------------------------------------------------------
# Category 3: Fake job offers
# -----------------------------------------------------------------
JOB_TEMPLATES = [
    "HONGERA! Umechaguliwa kwa kazi ya {company}. Mshahara TSh {amount}. Tuma {fee} kwa {phone} kupokea fomu",
    "NAFASI ZA AJIRA {company} zinapatikana. Tuma CV na malipo ya usajili TSh {fee} kwa {operator} {phone}",
    "Kazi mpya {company}! Mshahara mzuri. Lipa ada ya usajili TSh {fee} kwa namba {phone} jina {name}",
    "TANGAZO: {company} inahitaji wafanyakazi. Tuma TSh {fee} kwa {phone} ili upate fomu ya maombi",
    "Umekubaliwa kazi {company}. Lipa ada ya mafunzo TSh {fee} kwa {operator} {phone} kabla ya kesho",
    "Habari njema! Maombi yako kwa {company} yamekubaliwa. Tuma TSh {fee} kwa {phone} kuthibitisha",
]

COMPANIES = ["TANESCO", "TRA", "Vodacom Tanzania", "NMB Bank", "CRDB",
             "DAWASA", "TANROADS", "Kampuni ya Madini", "Hospitali ya Taifa",
             "Halmashauri ya Jiji", "Wizara ya Afya", "Polisi"]

JOB_FEES = ["5000", "10000", "15000", "20000", "25000", "30000", "50000"]

# -----------------------------------------------------------------
# Category 4: Prize / lottery scams
# -----------------------------------------------------------------
PRIZE_TEMPLATES = [
    "HONGERA! Umeshinda zawadi ya TSh {amount} kutoka {sponsor}. Piga {phone} sasa kupokea",
    "MSHINDI WA WIKI: Namba yako imechaguliwa kushinda TSh {amount}! Bonyeza link au piga {phone}",
    "Tuna habari njema: umeshinda gari kutoka {sponsor}. Lipa kodi TSh {fee} kwa {operator} {phone}",
    "CONGRATULATIONS! You have won TSh {amount} from {sponsor}. Call {phone} to claim",
    "Namba yako ya simu imeshinda TSh {amount}. Tuma TSh {fee} kwa {phone} kuthibitisha",
    "ZAWADI MAALUM! Umechaguliwa kupokea TSh {amount}. Wasiliana {phone} {urgency}",
    "Ushinde mkubwa! TSh {amount} kutoka {sponsor}. Lipa ada ndogo ya TSh {fee} kwa {phone}",
]

SPONSORS = ["Vodacom", "Airtel", "Tigo", "M-Pesa", "TBL", "Coca-Cola Tanzania",
            "Azam TV", "Bet9ja", "Premier Bet", "DStv"]

# -----------------------------------------------------------------
# Category 5: Family emergency / authority impersonation
# -----------------------------------------------------------------
EMERGENCY_TEMPLATES = [
    "Mjukuu wangu, niko hospitalini. Nakuhitaji utume {amount} {urgency} kwa {operator} {phone}",
    "Baba/Mama, nimepata ajali. Tafadhali tuma {amount} kwa daktari kwa {phone} ({operator})",
    "Mzee, nimeibiwa pesa zote. Tuma {amount} kwenye {operator} {phone} jina {name} {urgency}",
    "Habari, ni mimi ndugu yako. Nina shida kubwa, nahitaji {amount} kwa {operator} {phone}",
    "POLISI: Mtoto wako amekamatwa. Lipa dhamana TSh {amount} kwa {phone} kabla ya kesho",
    "TANESCO: Akaunti yako ya umeme itazimwa. Lipa TSh {fee} kwa {phone} sasa",
    "NIDA: Kitambulisho chako kitafutwa. Hakikisha taarifa zako kwa kupiga {phone}",
    "Benki: Akaunti yako imezuiwa. Piga {phone} kuthibitisha taarifa zako za siri",
    "Mjukuu wangu ndugu niliyokukabizi hiyo uwe makini na pesa hizo zinazokuja, tuma {amount} kwa {phone}",
]

# -----------------------------------------------------------------
# Generation
# -----------------------------------------------------------------
def fill_template(template: str) -> str:
    return template.format(
        amount=random.choice(AMOUNTS),
        fee=random.choice(JOB_FEES),
        operator=random.choice(MOBILE_OPERATORS),
        phone=random.choice(PHONE_NUMBERS),
        phone2=random.choice(PHONE_NUMBERS),
        name=random.choice(SWAHILI_NAMES),
        urgency=random.choice(URGENCY_WORDS),
        company=random.choice(COMPANIES),
        sponsor=random.choice(SPONSORS),
    )


def generate_category(templates, category_name, n_per_template=8):
    """Generate variations from each template."""
    messages = []
    for tmpl in templates:
        for _ in range(n_per_template):
            messages.append({
                "message": fill_template(tmpl),
                "label": "phishing",
                "source": f"synthetic_{category_name}",
            })
    return messages


def main():
    all_messages = []
    all_messages += generate_category(MOBILE_MONEY_TEMPLATES, "mobile_money", 10)
    all_messages += generate_category(FREEMASON_TEMPLATES, "freemason", 8)
    all_messages += generate_category(JOB_TEMPLATES, "fake_job", 8)
    all_messages += generate_category(PRIZE_TEMPLATES, "prize", 8)
    all_messages += generate_category(EMERGENCY_TEMPLATES, "emergency", 8)

    df = pd.DataFrame(all_messages)
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    print("=" * 60)
    print("SYNTHETIC PHISHING GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total messages generated: {len(df)}")
    print(f"\nPer-category breakdown:")
    print(df["source"].value_counts().to_string())
    print(f"\nMessage length stats:")
    print(df["message"].str.len().describe().round(1).to_string())

    print(f"\n5 random samples:")
    for i, row in df.sample(5, random_state=1).iterrows():
        print(f"\n  [{row['source']}]")
        print(f"  {row['message'][:200]}")

    out_path = PROCESSED_DIR / "synthetic_phishing.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"[OK] Saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
