import re
import sys
import os


class parse_log:

    def __init__(self, outfile):

        with open (outfile) as output_line:
            self.lines = output_line.readlines()

    def Check_SCF(self):

        scf_state = True

        for ii in range(len(self.lines)):
            ll = self.lines[ii]
            if (re.match(r"\s*SCF IS UNCONVERGED",ll)):
                scf_state = "scf error"

        return scf_state

    def Check_VIB(self):

        vib_state = True

        for ii in range(len(self.lines)):
            ll = self.lines[ii]
            if (re.search(r"\s*THE VIBRATIONAL ANALYSIS IS NOT VALID !!!",ll)):
                vib_state = "Not SP"

        return vib_state

        
    def getNumberElectron(self):

        for ii in range(len(self.lines)):
            ll = self.lines[ii]

            mmat = re.search(r"^[\s]*NUMBER OF OCCUPIED ORBITALS[^=]+=[\s]*([^\s]+)",ll)
            if(mmat):
                if(re.search("ALPHA",ll)):
                    num_occu_alpha = int(mmat.group(1))
                if(re.search("BETA",ll)):
                    num_occu_beta = int(mmat.group(1))

        return num_occu_alpha, num_occu_beta

    def getEnergy(self):

        energy = []

        for ii in range(len(self.lines)):
            ll = self.lines[ii]
       
            if(re.search(r"^[\s]*FINAL", ll)):
                #print(ll)
                sline = ll.split()
                energy.append(sline[-4])

        Comp_SS, Ideal_SS = self.Estimate_SpinDiff()

        Energy_spin = [float(energy[-1]), Comp_SS-Ideal_SS]

        #return float(energy[-1])
        return Energy_spin

    def extract_inputChargeSpin(self):
        charge = 0
        spinmulti  = 1

        for ii in range(len(self.lines)):
            ll = self.lines[ii]
            if(re.search("CHARGE OF MOLECULE",ll)):
                #print(ll)
                csline = ll.split()
                charge = float(csline[-1])
            if(re.search("SPIN MULTIPLICITY", ll)):
                #print(ll)
                msline = ll.split()
                spinmulti = float(msline[-1])

        return charge, spinmulti

    def Estimate_SpinDiff(self):

        SpinSqu = []

        InputCharge, InputSpinMulti = self.extract_inputChargeSpin()

        for ii in range(len(self.lines)):
            ll = self.lines[ii]
            if(re.search(r"^[\s]*S-SQUARED", ll)):
                #print(ll)
                sline = ll.split()
                SpinSqu.append(sline[-1])
            else:
                SpinSqu.append(0.0)

        Computed_SS = float(SpinSqu[-1]) 

        TotalS = (InputSpinMulti-1) / 2
        Ideal_SS = TotalS * (TotalS+1)

        return Computed_SS, Ideal_SS

    def getTDDFT(self):

        Wavel = []
        OS = []
        flag = 0
        ii = 0
        while(ii < len(self.lines)):
            ll = self.lines[ii]
        
            if (flag == 0 and re.search("SUMMARY OF TDDFT RESULTS", ll)):
                flag = 1
                Wavel = []
                OS = []
                ii += 2
            if (flag == 1 and re.match(r"\s+[0-9]",ll) and not re.search('HARTREE', ll)):
                sline = ll.split()
                if len(sline) > 4: 
                    Wavel.append(1240/float(sline[-5]))
                    OS.append(float(sline[-1]))
            if (flag == 1 and  re.match(r"\s+\n", ll)):
                flag = 0
            ii += 1
        
        return Wavel, OS

    def getChargeSpin(self):

        Element = []
        Mulliken_charge = []
        Lowdin_charge = []
        Spin = []
        flag = 0 
        ii = 0
        while(ii < len(self.lines)):
            ll = self.lines[ii]
            if (flag == 0 and re.search("TOTAL MULLIKEN AND LOWDIN ATOMIC POPULATIONS", ll)):
                flag = 1
                Element = []
                Mulliken_charge = []
                Lowdin_charge = []
                ii += 1
            if (flag == 1 and re.match(r"\s+[0-9]",ll) and not re.search('HARTREE', ll)):
                sline = ll.split()
                if len(sline) > 4: 
                    Element.append(sline[1])
                    Mulliken_charge.append(float(sline[3]))
                    Lowdin_charge.append(float(sline[5]))
            if (flag == 1 and re.match(r"\s*\n", ll)):
                flag = 0
            if (flag == 0 and re.search("ATOMIC SPIN DENSITY AT THE NUCLEUS", ll)):
                flag = 2
                Spin = []
                ii += 1
            if (flag == 2 and re.match(r"\s+[0-9]",ll) and not re.search('HARTREE', ll)):
                sline = ll.split()
                if len(sline) > 4: 
                    Spin.append(float(sline[3]))
            if (flag == 2 and re.match(r"\s*\n", ll)):
                flag = 0

            ii += 1
        
        return Element, Mulliken_charge, Spin

    def getBlock(self, label):
        flag = 0
        ret = []
        currentlist = []
        ii = 0
        while(ii < len(self.lines)):
            ll = self.lines[ii]
            
            if(flag == 1 and (re.search(r"^[\s]*----",ll) or re.search(r"\.\.\.\.\.\.",ll))):
                flag = 0
                ret.append(currentlist)
                currentlist = []
                
            if(re.search(r"^[\s]*"+label,ll)):
                if(re.search(r"^[\s]*----",self.lines[ii-1]) and re.search(r"^[\s]*----",self.lines[ii+1])):
                    flag = 1
                    ii += 2
                    continue
                    
            if(flag == 1):
                currentlist.append(ll)
            
            ii+=1
        if(len(currentlist)):
            ret.append(currentlist)
        return ret
    
    def getMO_set(self, block):
        flag = 0
        ret = []
        currentlist = []
        ii = 0
        
        hitflag = 0
        
        elec_flag = 0
        alpha_indices = []
        alpha_values = []
        beta_indices = []
        beta_values = []
        while(ii < len(block)-1):
            ll = block[ii]
            nex = block[ii+1]
            if(re.search(r"ALPHA SET",ll)):
                #print (ll)
                elec_flag = 1
            if(re.search(r"BETA SET",ll)):
                #print (ll)
                elec_flag = -1
            if(re.search(r"^          +[0-9]+ ",ll) and re.search(r"^          ",nex) and not re.search("[A-DF-Za-df-z]",nex)):
                ipt = re.split(r"[\s]+",re.sub(r"[\s]+$","",re.sub(r"^[\s]+","",ll)))
                vpt = re.split(r"[\s]+",re.sub(r"[\s]+$","",re.sub(r"^[\s]+","",nex)))
                for pp in range(len(ipt)):
                    if elec_flag == 1:
                        alpha_indices.append(int(ipt[pp]))
                        alpha_values.append(float(vpt[pp]))
                    if elec_flag == -1:
                        beta_indices.append(int(ipt[pp]))
                        beta_values.append(float(vpt[pp]))
            ii += 1
        
        #print(alpha_indices)
        #print(alpha_values)
        #print(beta_indices)
        #print(beta_values)
        
        if(len(alpha_indices) != len(alpha_values) or len(beta_indices) != len(beta_values)):
            raise Exception("???different length bet index list and value list. parsing error???")
        
        return alpha_values, beta_values

    def getMO_single(self, block):
        ret = []
        currentlist = []
        ii = 0
        
        hitflag = 0
        
        elec_flag = 0
        indices = []
        values = []
        while(ii < len(block)-1):
            ll = block[ii]
            nex = block[ii+1]
            if(re.search("^          +[0-9]+ ",ll) and re.search("^          ",nex) and not re.search("[A-DF-Za-df-z]",nex)):
                ipt = re.split(r"[\s]+",re.sub(r"[\s]+$","",re.sub(r"^[\s]+","",ll)))
                vpt = re.split(r"[\s]+",re.sub(r"[\s]+$","",re.sub(r"^[\s]+","",nex)))
                for pp in range(len(ipt)):
                        indices.append(int(ipt[pp]))
                        values.append(float(vpt[pp]))
            ii += 1
        
        #print(indices)
        #print(values)
        
        if(len(indices) != len(values)) :
            raise Exception("???different length bet index list and value list. parsing error???")
        
        return values

    def gethomolumogap(self, alpha_values, beta_values, num_alpha_elec, num_beta_elec):
    
        ret1 = None
        ret2 = None
        for ii in range(len(alpha_values)):
            if(ii == num_alpha_elec-1):
                ret1 = alpha_values[ii]
                ret2 = alpha_values[ii+1]
        alpha_gap  =(float(ret2)-float(ret1))*27.211
        
        for ii in range(len(beta_values)):
            if(ii == num_beta_elec-1):
                ret1 = beta_values[ii]
                ret2 = beta_values[ii+1]
        beta_gap  =(float(ret2)-float(ret1))*27.211
        
        return  alpha_gap, beta_gap

    def getDipoleMoment(self, block):
        ii = 0
        ret = None
        while(ii < len(block)-1):
            ll = block[ii]
            nex = block[ii+1]
            #/D/
            if(re.search(r"/D/",ll)):
                ipt = re.split(r"[\s]+",re.sub(r"[\s]+$","",re.sub(r"^[\s]+","",ll)))
                vpt = re.split(r"[\s]+",re.sub(r"[\s]+$","",re.sub(r"^[\s]+","",nex)))
                
                for pp in range(len(ipt)):
                    if(ipt[pp] == r"/D/"):#/D/
                        ret =float(vpt[pp])
            ii += 1
        #if(ret == None):
        #    raise Exception("??? data pasing error?"+"\n".join(block));
    
        return ret

    def getFreq(self, block):
        ii = 0 
        Frequency = []
        IR = []
        Raman = []

        while(ii < len(block)-1):
            ll = block[ii]
            if(re.search(r"\s* FREQUENCY:",ll)):
                iFreq = re.split(r"[\s]+",re.sub(r"[\s]+$","",re.sub(r"^[\s]+","",ll)))
                #print(iFreq[1:])
                for pp in range(1, len(iFreq)):
                    Frequency.append(float(iFreq[pp]))

            if(re.search(r"\s* IR INTENSITY:",ll)):
                iIR = re.split(r"[\s]+",re.sub(r"[\s]+$","",re.sub(r"^[\s]+","",ll)))
                #print(iInt[2:])
                for pp in range(2, len(iIR)):
                    IR.append(float(iIR[pp]))

            if(re.search(r"\s* RAMAN ACTIVITY:",ll)):
                iRaman = re.split(r"[\s]+",re.sub(r"[\s]+$","",re.sub(r"^[\s]+","",ll)))
                #print(iInt[2:])
                for pp in range(2, len(iRaman)):
                    Raman.append(float(iRaman[pp]))
            ii += 1

        return Frequency[6:], IR[6:], Raman[6:]

    def getThermo(self, block):
        ii = 0
        E_0 = []
        U = []
        H = []
        G = []
        Cv = []
        Cp = []
        S = []

        flag = 0

        while(ii < len(block)-1):
            ll = block[ii]
            if(flag == 0 and re.search(r"\s* THE HARMONIC ZERO POINT ENERGY IS",ll)):
                flag = 1
                #print (ll)
                pass
            if (flag == 1 and re.match(r"\s+[0-9]",ll) and not re.search('WORDS', ll)):
                #print (ll)
                sline = ll.split()
                if len(sline) > 2: 
                    E_0.append(float(sline[0]))
                    E_0.append(float(sline[2]))
                #print (E_0)

            if(flag == 1 and re.search(r"\bKJ/MOL\b\s+\bKJ/MOL\b\s+\bKJ/MOL\b\s+\bJ/MOL-K\b\s+\bJ/MOL-K\b\s+\bJ/MOL-K\b",ll)):
                flag = 2
                #print (ll)

            if (flag == 2 and re.match(r"\s+TOTAL",ll)):
                #print (ll)
                sline = ll.split()
                if len(sline) > 4: 
                    U.append(float(sline[1]))
                    H.append(float(sline[2]))
                    G.append(float(sline[3]))
                    Cv.append(float(sline[4]))
                    Cp.append(float(sline[5]))
                    S.append(float(sline[6]))

            if(flag == 2 and re.search(r"\bKCAL/MOL\b\s+\bKCAL/MOL\b\s+\bKCAL/MOL\b\s+\bCAL/MOL-K\b\s+\bCAL/MOL-K\b\s+\bCAL/MOL-K\b" ,ll)):
                flag = 3
                #print (ll)

            if (flag == 3 and re.match(r"\s* TOTAL",ll) and not re.search('WALL', ll)):
                #print (ll)
                sline = ll.split()
                if len(sline) > 4: 
                    U.append(float(sline[1]))
                    H.append(float(sline[2]))
                    G.append(float(sline[3]))
                    Cv.append(float(sline[4]))
                    Cp.append(float(sline[5]))
                    S.append(float(sline[6]))

            ii += 1

        return E_0, U, H, G, Cv, Cp, S

if __name__ == '__main__':
    usage ='Usage; %s infile' % sys.argv[0]

    try:
        infilename = sys.argv[1]
    except:
        print (usage); sys.exit()

    parselog = parse_log(infilename)
    print(parselog.Check_SCF())

    print(parselog.Check_VIB())

    flag_homolumo = False
    flag_dipole = False

    num_occu_alpha, num_occu_beta  = parselog.getNumberElectron()

    print(parselog.getEnergy())

#    print(parselog.Estimate_SpinDiff())


    Wavel, OS = parselog.getTDDFT()

    print ("Wave length:", Wavel)
    print ("OS:", OS)
    

    Element, MullCharge, Spin = parselog.getChargeSpin()

    print(Element, MullCharge, Spin)

    bb = parselog.getBlock("EIGENVECTORS")
    #print(bb[-2])
    #print(bb[-1])
    alpha_values = parselog.getMO_single(bb[-2])
    beta_values = parselog.getMO_single(bb[-1])
    #bb = getBlock(llines,"MOLECULAR ORBITALS")
    #alpha_values, beta_values = getMO_set(bb[-1])
    alpha_gap, beta_gap = parselog.gethomolumogap(alpha_values, beta_values, num_occu_alpha, num_occu_beta)
    dd = parselog.getBlock("ELECTROSTATIC MOMENTS")    
    #dd = parselog.getBlock("SYSTEM ELECTROSTATIC MOMENTS")    
    print(dd)
    dval = parselog.getDipoleMoment(dd[-1])
    #print("HOMO:\t"+str(hval[0]))
    #print("LUMO:\t"+str(hval[1]))
    print("HOMO - LUMO gap:\t"+str(alpha_gap))
    print("HOMO - LUMO gap:\t"+str(beta_gap))
    print("DIPOLEMOMENT:\t"+str(dval))

    ff = parselog.getBlock("NORMAL COORDINATE ANALYSIS IN THE HARMONIC APPROXIMATION")    
    Freq, IR, Raman = parselog.getFreq(ff[0])
    print (f'Freq: {Freq}')
    print (f'IR: {IR}')
    print (f'Raman: {Raman}')

    tt = parselog.getBlock("THERMOCHEMISTRY AT ")    
    print(parselog.getThermo(tt[0]))



