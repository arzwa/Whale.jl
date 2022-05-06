using Whale, NewickTree, Plots

# LCA reconciliation
# ==================
# Note that we need to root the trees...

S = nw"(((A:0.3,B:0.3):0.5,(C:0.6,D:0.6):0.2):0.2,E:1.);"
G = nw"((((A_5,B_6),D_9),((A_12,B_13),((C_16,C_17),D_18))),(E_20,E_21));"
R = Whale.lca_reconciliation(S, G)

plot(R, transform=true)

S = nw"((mpo:4.9058224222,ppa:4.9058224222):0.41709669779999997,(smo:4.3125139636,((scu:0.9079999999999999,afi:0.9079999999999999):3.1100588649999996,(((vvi:1.1702839392,bvu:1.1702839392):0.6385731165,atr:1.8088570558000001):1.3184042224,(cpa:3.0295,(gbi:2.9556845545,sgi:2.9556845545):0.0738494371):0.0977272865):0.8907975869000001):0.2944550986):1.0104051564);"
G1 = nw"(scu_scu009750,((scu_scu016997,afi_afi011856),(((cpa_cpa023318,(((((ppa_ppa024310,ppa_ppa022959),ppa_ppa004716),((smo_smo017511,smo_smo015211),mpo_mpo001366)),(gbi_gbi026386,(((sgi_sgi028155,sgi_sgi024730),(sgi_sgi032268,(sgi_sgi014833,sgi_sgi001840))),(cpa_cpa027835,cpa_cpa027808)))),(((gbi_gbi040481,gbi_gbi008651),gbi_gbi008650),cpa_cpa001608))),(gbi_gbi029755,(sgi_sgi036007,(bvu_bvu006454,(vvi_vvi011478,((vvi_vvi014195,bvu_bvu009113),atr_atr012399)))))),atr_atr001985)),afi_afi010299);"
G2 = nw"(scu_scu009750,((scu_scu016997,afi_afi011856),(((((((ppa_ppa024310,ppa_ppa022959),ppa_ppa004716),((smo_smo017511,smo_smo015211),mpo_mpo001366)),(gbi_gbi026386,(((sgi_sgi028155,sgi_sgi024730),(sgi_sgi032268,(sgi_sgi014833,sgi_sgi001840))),(cpa_cpa027835,cpa_cpa027808)))),(((gbi_gbi040481,gbi_gbi008651),gbi_gbi008650),cpa_cpa001608)),(cpa_cpa023318,(gbi_gbi029755,(sgi_sgi036007,(bvu_bvu006454,(vvi_vvi011478,((vvi_vvi014195,bvu_bvu009113),atr_atr012399))))))),atr_atr001985)),afi_afi010299);"
G3 = nw"(scu_scu009750,((scu_scu016997,afi_afi011856),(((cpa_cpa023318,((((ppa_ppa024310,ppa_ppa022959),ppa_ppa004716),((smo_smo017511,smo_smo015211),mpo_mpo001366)),((gbi_gbi026386,(((sgi_sgi028155,sgi_sgi024730),(sgi_sgi032268,(sgi_sgi014833,sgi_sgi001840))),(cpa_cpa027835,cpa_cpa027808))),(((gbi_gbi040481,gbi_gbi008651),gbi_gbi008650),cpa_cpa001608)))),(gbi_gbi029755,(sgi_sgi036007,(bvu_bvu006454,(vvi_vvi011478,((vvi_vvi014195,bvu_bvu009113),atr_atr012399)))))),atr_atr001985)),afi_afi010299);"

G1 = set_outgroup(getlca(G1, "ppa_ppa024310", "mpo_mpo001366"))
G2 = set_outgroup(getlca(G2, "ppa_ppa024310", "mpo_mpo001366"))
G3 = set_outgroup(getlca(G3, "ppa_ppa024310", "mpo_mpo001366"))

R1 = Whale.lca_reconciliation(S, G1)
R2 = Whale.lca_reconciliation(S, G2)
R3 = Whale.lca_reconciliation(S, G3)

ps = map([R1, R2, R3]) do R
    plot(R, transform=true, right_margin=40Plots.mm, size=(300,900))
end

plot(ps..., layout=(1,3), size=(900,900))

trees = [
    nw"(scu_scu005945,((smo_smo007306,(((ppa_ppa029032,ppa_ppa024492),ppa_ppa013346),mpo_mpo000996)),(sgi_sgi034879,(cpa_cpa004988,gbi_gbi019232))),afi_afi003307);",
    nw"(scu_scu005945,((smo_smo007306,(((ppa_ppa029032,ppa_ppa024492),ppa_ppa013346),mpo_mpo000996)),(cpa_cpa004988,(sgi_sgi034879,gbi_gbi019232))),afi_afi003307);",
    nw"(scu_scu005945,((smo_smo007306,((ppa_ppa024492,(ppa_ppa029032,ppa_ppa013346)),mpo_mpo000996)),(sgi_sgi034879,(cpa_cpa004988,gbi_gbi019232))),afi_afi003307);"
   ]
Gs = map(x->set_outgroup(getlca(x, "ppa_ppa029032", "mpo_mpo000996")), trees)
Rs = map(x->Whale.lca_reconciliation(S, x), Gs)
ps = map(Rs) do R
    plot(R, transform=true, right_margin=40Plots.mm, size=(300,900))
end
plot(ps..., layout=(1,length(ps)), size=(900,500))

trees = [
 nw"((scu_scu014053,afi_afi008399),((smo_smo006491,(ppa_ppa028258,mpo_mpo008993)),(((sgi_sgi017945,(sgi_sgi021898,(sgi_sgi007132,sgi_sgi005323))),((sgi_sgi017279,(sgi_sgi032786,sgi_sgi012343)),(gbi_gbi009329,cpa_cpa009713))),((vvi_vvi025159,(bvu_bvu022168,bvu_bvu012234)),atr_atr018441))),afi_afi008398);",
 nw"((scu_scu014053,afi_afi008399),((smo_smo006491,(ppa_ppa028258,mpo_mpo008993)),(((sgi_sgi017279,(sgi_sgi032786,sgi_sgi012343)),((sgi_sgi017945,(sgi_sgi021898,(sgi_sgi007132,sgi_sgi005323))),(gbi_gbi009329,cpa_cpa009713))),((vvi_vvi025159,(bvu_bvu022168,bvu_bvu012234)),atr_atr018441))),afi_afi008398);",
 nw"((scu_scu014053,afi_afi008399),((smo_smo006491,(ppa_ppa028258,mpo_mpo008993)),((((sgi_sgi017279,(sgi_sgi032786,sgi_sgi012343)),(sgi_sgi017945,(sgi_sgi021898,(sgi_sgi007132,sgi_sgi005323)))),(gbi_gbi009329,cpa_cpa009713)),((vvi_vvi025159,(bvu_bvu022168,bvu_bvu012234)),atr_atr018441))),afi_afi008398);"
]
Gs = map(x->set_outgroup(getlca(x, "ppa_ppa028258", "mpo_mpo008993")), trees)
Rs = map(x->Whale.lca_reconciliation(S, x), Gs)
ps = map(Rs) do R
    plot(R, transform=true, right_margin=40Plots.mm, size=(300,900))
end
plot(ps..., layout=(1,length(ps)), size=(900,500))

trees = [
 nw"(scu_scu009538,((smo_smo003971,(smo_smo014264,mpo_mpo004073)),((sgi_sgi009282,cpa_cpa030842),((gbi_gbi033789,(cpa_cpa027118,(cpa_cpa021890,cpa_cpa002861))),((sgi_sgi009888,ppa_ppa018259),(((vvi_vvi009437,bvu_bvu018350),atr_atr014962),(((vvi_vvi013619,vvi_vvi011630),(vvi_vvi003562,vvi_vvi003188)),bvu_bvu012323)))))),afi_afi014873);",
 nw"(scu_scu009538,((smo_smo014264,smo_smo003971),(mpo_mpo004073,((sgi_sgi009282,cpa_cpa030842),((gbi_gbi033789,(cpa_cpa027118,(cpa_cpa021890,cpa_cpa002861))),((sgi_sgi009888,ppa_ppa018259),(((vvi_vvi009437,bvu_bvu018350),atr_atr014962),(((vvi_vvi013619,vvi_vvi011630),(vvi_vvi003562,vvi_vvi003188)),bvu_bvu012323))))))),afi_afi014873);",
 nw"(scu_scu009538,(smo_smo003971,((smo_smo014264,mpo_mpo004073),((sgi_sgi009282,cpa_cpa030842),((gbi_gbi033789,(cpa_cpa027118,(cpa_cpa021890,cpa_cpa002861))),((sgi_sgi009888,ppa_ppa018259),(((vvi_vvi009437,bvu_bvu018350),atr_atr014962),(((vvi_vvi013619,vvi_vvi011630),(vvi_vvi003562,vvi_vvi003188)),bvu_bvu012323))))))),afi_afi014873);"]
Gs = map(x->set_outgroup(getlca(x, "mpo_mpo004073")), trees)
Rs = map(x->Whale.lca_reconciliation(S, x), Gs)
ps = map(Rs) do R
    plot(R, transform=true, right_margin=40Plots.mm, size=(300,900))
end
plot(ps..., layout=(1,length(ps)), size=(900,500))

trees = [
 nw"(scu_scu009622,((smo_smo014600,mpo_mpo004146),((sgi_sgi025063,sgi_sgi022147),(((cpa_cpa028946,(gbi_gbi005153,cpa_cpa021755)),((sgi_sgi013473,gbi_gbi013945),(cpa_cpa023350,cpa_cpa019958))),((vvi_vvi004386,atr_atr008721),((vvi_vvi000333,bvu_bvu020727),atr_atr014153))))),afi_afi001348);",
 nw"(scu_scu009622,((smo_smo014600,mpo_mpo004146),((sgi_sgi025063,sgi_sgi022147),((cpa_cpa028946,((gbi_gbi005153,cpa_cpa021755),((sgi_sgi013473,gbi_gbi013945),(cpa_cpa023350,cpa_cpa019958)))),((vvi_vvi004386,atr_atr008721),((vvi_vvi000333,bvu_bvu020727),atr_atr014153))))),afi_afi001348);",
 nw"(scu_scu009622,((smo_smo014600,mpo_mpo004146),((sgi_sgi025063,sgi_sgi022147),(((cpa_cpa028946,(gbi_gbi005153,cpa_cpa021755)),(cpa_cpa023350,((sgi_sgi013473,gbi_gbi013945),cpa_cpa019958))),((vvi_vvi004386,atr_atr008721),((vvi_vvi000333,bvu_bvu020727),atr_atr014153))))),afi_afi001348);",
 nw"(scu_scu009622,((smo_smo014600,mpo_mpo004146),((sgi_sgi025063,sgi_sgi022147),((cpa_cpa028946,((gbi_gbi005153,cpa_cpa021755),(cpa_cpa023350,((sgi_sgi013473,gbi_gbi013945),cpa_cpa019958)))),((vvi_vvi004386,atr_atr008721),((vvi_vvi000333,bvu_bvu020727),atr_atr014153))))),afi_afi001348);",
 nw"(scu_scu009622,((smo_smo014600,mpo_mpo004146),((sgi_sgi025063,sgi_sgi022147),(((gbi_gbi005153,cpa_cpa021755),(cpa_cpa028946,((sgi_sgi013473,gbi_gbi013945),(cpa_cpa023350,cpa_cpa019958)))),((vvi_vvi004386,atr_atr008721),((vvi_vvi000333,bvu_bvu020727),atr_atr014153))))),afi_afi001348);",
 nw"(scu_scu009622,(mpo_mpo004146,(smo_smo014600,((sgi_sgi025063,sgi_sgi022147),(((cpa_cpa028946,(gbi_gbi005153,cpa_cpa021755)),((sgi_sgi013473,gbi_gbi013945),(cpa_cpa023350,cpa_cpa019958))),((vvi_vvi004386,atr_atr008721),((vvi_vvi000333,bvu_bvu020727),atr_atr014153)))))),afi_afi001348);"]
Gs = map(x->set_outgroup(getlca(x, "mpo_mpo004146")), trees)
Rs = map(x->Whale.lca_reconciliation(S, x), Gs)
ps = map(Rs) do R
    plot(R, transform=true, right_margin=40Plots.mm, size=(300,900))
end
plot(ps..., layout=(2,3), size=(900,900))



