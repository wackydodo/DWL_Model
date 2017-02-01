from __future__ import division
import numpy as np
np.random.seed(1)

pindyck_tmp_k = [2.81, 4.6134, 6.14]
pindyck_tmp_theta = [1.6667, 1.5974, 1.53139]
pindyck_tmp_displace = [-0.25,  -0.5,  -1.0]
disaster_tail = 13.0
pindyck_impact_k = 4.5
pindyck_impact_theta = 21341.0
pindyck_impact_displace = -0.0000746
maxh = 100

#my tree
growth = .02
peak_tmp = 9.0
peak_tmp_interval = 30.
nperiods = 6
decision_times = [ 0, 15, 45, 85, 185, 285, 385]
final_states = 32
probs = [1/32] * 32

def gammaArray(shape, rate, dimension):
    scale = 1/rate
    y = np.random.gamma(shape, scale, dimension)
    return y

def normalArray(mean, stdev, dimension):
    y = np.random.normal(mean, stdev, dimension)
    return y

def uniformArray(dimension):
    y = np.random.random(dimension)
    return y


def tmperature_interpolation(final_tmperature, draw_number, nperiods, maxh):

    tmperature_mesh = map(lambda(x): [2 * x * (1 - 0.5**(decision_times[p+1]/maxh)) for p in range(nperiods)], final_tmperature)
    tmperature_mesh = np.array(tmperature_mesh)
    return tmperature_mesh

def consumption_calculation(impact, final_tmperature, decision_times, maxh, draw_number):
    end_time = decision_times[1:]
    term1 = -2.0 * impact * maxh * final_tmperature / np.log(0.5)
    term23 = [(growth - 2.0 * impact * final_tmperature) * time  + \
             ( 2.0 * impact * maxh * final_tmperature * 0.5 ** (time / maxh) ) / np.log(0.5) for time in end_time]
    term23 = np.array(term23)
    consump = np.exp(term1 + term23)
    return np.transpose(consump)

def add_tipping(disaster, disaster_consumption, tmp, consump, peak_tmp, decision_times):

    period_length = decision_times[1:] - decision_times[:-1]
    max_tmp = np.minimum(tmp, peak_tmp)
    # ave_prob_of_survival = 1. - (tmp/ max_tmp) ** 2
    # prob_of_survival_this_period = ave_prob_of_survival ** (period_length / peak_tmp_interval)
    # disaster_bar = prob_of_survival_this_period

    # if( disaster[counter,p] > disaster_bar and first_bp == 0 ):
 #                            for pp in range(p,self.my_tree.nperiods):
 #                                consump[counter,pp] = consump[counter,pp] * math.exp(-disaster_consumption[counter])
 #                                first_bp = 1

    return consump
    


def simulation(draw_number, decision_times=[ 0, 15, 45, 85, 185, 285, 385], nperiods=6, maxh=100):
    decision_times = np.array(decision_times)
    
    # 450 GHG level, pindyck mapping

    # tmperature
    tmperature = gammaArray(pindyck_tmp_k[0], pindyck_tmp_theta[0], draw_number) + pindyck_tmp_displace[0]
    tmperature[tmperature < 0] = 0
    tmp = tmperature_interpolation(tmperature, draw_number, nperiods, maxh)


    #consumption
    impact = gammaArray(pindyck_impact_k, pindyck_impact_theta, draw_number) + pindyck_impact_displace
    consump = consumption_calculation(impact, tmperature, decision_times, maxh, draw_number)

    #tipping point
    disaster = uniformArray([draw_number, nperiods])
    disaster_consumption = gammaArray(1.0, disaster_tail, draw_number)
    consump = add_tipping(disaster, disaster_consumption, tmp, consump, peak_tmp, decision_times)
    
    #sort consumption
    peak_con = np.exp(growth * decision_times[1:])
    consump = consump[consump[:,nperiods-1].argsort()]
    tmp = tmp[ tmp[:,nperiods-1].argsort()]
    damage= 1. - consump/peak_con

    #calculate sorted damage
    d = np.zeros((final_states, nperiods))
    firstob = 0
    lastob = int(probs[0]*(draw_number-1))
    for n in np.arange(0, final_states):
        d[n,:] = np.maximum(damage[range(firstob,lastob), :].mean(axis=0), 0)
        firstob = lastob + 1
        if( n < final_states - 1 ):
            lastob = int(sum(probs[0:n+2]) * (draw_number-1)-1)
    return d


if __name__ == "__main__":

    print simulation(100).shape