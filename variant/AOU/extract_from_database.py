import hail
import re

dt = hail.import_table("gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/aux/vat/vat_complete.bgz.tsv.gz", force_bgz = True, types={'position': hail.tint32})
dt = dt.key_by('contig','position','ref_allele','alt_allele').select("gvs_eur_an", "gvs_eur_ac", "gvs_all_an", "gvs_all_ac").distinct()
dt = dt.filter((dt['ref_allele'].length()==1) & (dt['alt_allele'].length()==1))

# should write to a cloud workspace
dt.write("f{cloud_location}/AOU_af_SNV.ht")

