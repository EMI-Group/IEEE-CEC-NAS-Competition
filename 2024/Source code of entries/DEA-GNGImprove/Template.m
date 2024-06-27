function epsilon = Template(type)
% 参数组1
epsFe = 0.03;
epsF1h = 0.005;
epsF2h = 0.015;
epsFc = 0;

switch(type)
    case 'CitySegMOP1'
        epsilon = [epsFe;epsF1h];
    case 'CitySegMOP2'
        epsilon = [epsFe;epsF1h;epsFc];
    case 'CitySegMOP3'
        epsilon = [epsFe;epsF1h;epsFc];
    case 'CitySegMOP4'
        epsilon = [epsFe;epsF1h;epsF2h;epsFc];
    case 'CitySegMOP5'
        epsilon = [epsFe;epsF1h;epsF2h;epsFc;epsFc];
    case 'CitySegMOP6'
        epsilon = [epsFe;epsF1h];
    case 'CitySegMOP7'
        epsilon = [epsFe;epsF1h;epsFc];
    case 'CitySegMOP8'
        epsilon = [epsFe;epsF1h;epsFc];
    case 'CitySegMOP9'
        epsilon = [epsFe;epsF1h;epsF2h;epsFc];
    case 'CitySegMOP10'
        epsilon = [epsFe;epsF1h;epsF2h;epsFc;epsFc];
    case 'CitySegMOP11'
        epsilon = [epsFe;epsF1h;epsF1h];
    case 'CitySegMOP12'
        epsilon = [epsFe;epsF1h;epsF1h;epsF2h;epsF2h];
    case 'CitySegMOP13'
        epsilon = [epsFe;epsF1h;epsF1h;epsF2h;epsF2h;epsFc];
    case 'CitySegMOP14'
        epsilon = [epsFe;epsF1h;epsF1h;epsF2h;epsF2h;epsFc];
    case 'CitySegMOP15'
        epsilon = [epsFe;epsF1h;epsF1h;epsF2h;epsF2h;epsFc;epsFc];
end

end