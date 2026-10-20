# aien-train L278 section 8: Chinchilla-optimal training on one GB10.
# N = parameters, L = layers, h = hidden width (nearest GPT-3 shapes; 50M shape is the author's choice).
# D = 20 N tokens, F = 6 N D FLOPs, peak p = 207.7e12 (measured FP8 best case), bw = 273e9 B/s, T = 524288 tokens/step.
# State floor: 23 bytes/param moved once per step. Activation floor: 2 x 34 h L bytes/token (store all, written once, read once).
# Recompute floor: 2 x 2 h L bytes/token (keep only each layer input; section 5.2 method).
for row in "50e6 8 512" "125e6 12 768" "350e6 24 1024" "1e9 24 2048" "3e9 32 2560"; do
set -- $row
awk -v N=$1 -v L=$2 -v h=$3 'BEGIN{D=20*N; F=6*N*D; p=207.7e12; T=524288; bw=273e9;
 m40=F/(0.40*p); m25=F/(0.25*p); m55=F/(0.55*p);
 st=(D/T)*23*N/bw; as=D*2*34*h*L/bw; ar=D*2*2*h*L/bw;
 printf "N=%g L=%d h=%d D=%g F=%.4g | math hours: 25%%=%.4g 40%%=%.4g 55%%=%.4g | state floor h=%.4g steps=%.0f | act store-all h=%.4g (%.3g x math@40%%) | act recompute h=%.4g (%.3g x math@40%%)\n", N,L,h,D,F, m25/3600,m40/3600,m55/3600, st/3600,D/T, as/3600, as/m40, ar/3600, ar/m40}'
done
