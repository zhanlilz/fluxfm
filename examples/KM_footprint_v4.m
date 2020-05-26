function [KMoutput,cumff] = KM_footprint_v4(class_img,img_res,Tpos,zm,ts,ustar,WS,sv,WD,L,qcflag)

% Korman & Meixner footprint script
% based on script by Jakob Sievers (05/2013), revised and annotated by Christian Wille
% cf. Korman & Meixner (2001): An Analytical Footprint Model For Non-Neutral Stratification. BLM 99: 207-224.
%
% Syntax: KM_footprint(class_img,img_res,Tpos,zm,ts,ustar,WS,sv,WD,L)
%   class_img = classified image matrix
%               value 0 = pixel not classified
%               value 1 = class 1 present
%               value 2 = class 2 present
%               etc.
%   img_res = pixel size in meters
%   Tpos = tower pixel position [row,column] counting from upper left (NW) corner of image matrix
%   zm = EC measurement height in meters
%   ts = timestamp of turbulence data
%   ustar = friction velocity
%   WS = wind speed
%   sv = standard deviation of crosswind
%   WD = wind direction in degrees
%   L = Monin-Obukhov length 
%
% Outputs:
%   class_img_with_tower.png:  check if tower position is correct 
%   roughness_length.png:  check roughness length (median is used for all wind directions)
%   yyyy_cum_2D_footprints.mat:  yearly files with 2D cumulative footprint matrix
%   yyyy_footprint_results.mat:  yearly files with half-hourly footprint results with following columns: 
%       timestamp, ustar, WS, sv, WD, L, image contribution, class 1 contribution, class 2 contribution, etc.
%
% last changed: March 2019, cwille

procdate = datestr(now,30);

%% deal with input data 

% filtering of turbulence data
zm = zm(qcflag);
ts = ts(qcflag);
ustar = ustar(qcflag);
WS = WS(qcflag);
sv = sv(qcflag);
WD = WD(qcflag);
L = L(qcflag);

halfhours = length(ts);

% number of classes in classified image
numclass = max(class_img(:));

% image size in pixels (N-S and E-W dimensions)
[NSdim,EWdim] = size(class_img);

% create a grid of distance to tower with 'hor' spanning W->E positions and 'ver' spanning N->S positions
[hor,ver] = meshgrid((1:EWdim)-Tpos(2),(1:NSdim)-Tpos(1));

% apply pixel size of image in meters
hor = img_res*hor; ver = img_res*ver;


%% initialize parameters of footprint model - part 1
% v. Karman constant
k = 0.4;

% phi_m KM eq. 33 - stability function
phi_m = zeros(halfhours,1);
phi_m(L<0) = (1-16.*zm(L<0)./L(L<0)).^-0.25;
phi_m(L>=0) = 1+5.*zm(L>=0)./L(L>=0);

% phi_c KM eq. 34 - stability function
phi_c = zeros(halfhours,1);
phi_c(L<0) = (1-16.*zm(L<0)./L(L<0)).^-0.5;
phi_c(L>=0) = 1+5.*zm(L>=0)./L(L>=0);

% psi_m KM eq. 35 - diabatic integration of the wind profile (using 1/phi_m as zeta)
psi_m = zeros(halfhours,1);
psi_m(L<0) = -2.*log(0.5*(1+1./phi_m(L<0)))-log(0.5*(1+(1./phi_m(L<0)).^2))+2*atan(1./phi_m(L<0))-pi/2;
psi_m(L>=0) = 5.*zm(L>=0)./L(L>=0);


%% evaluate roughness length z0 - moving window 0-360°

% roughness length z0 - original eqn. from Jakob - this seems wrong
% z0 = zm./(exp((WS*k)./ustar)+exp(psi_m));
% our own solution for z0; KM p.216
z0 = zm.*exp(psi_m-(k*WS./ustar));

% remove outliers
z0(z0>1000) = NaN;

% window of 45° moving in steps 0f 1°
halfbinwidth = 22;
z0med = nan(size(z0));
for kk = 0 : 359
    WDwrapped = WD;
    if kk<90
        WDwrapped(WD>270) = WD(WD>270) - 360;
    elseif kk>270
        WDwrapped(WD<90) = WD(WD<90) + 360;
    end
    idx1 = WD >= kk & WD < (kk+1);        
    idx2 = WDwrapped >= (kk-halfbinwidth) & WDwrapped < (kk+1+halfbinwidth);
    z0med(idx1) = nanmedian(z0(idx2));
end

figure('Units','normalized','OuterPosition',[0 0 1 1]);
polarplot(deg2rad(90-WD),z0,'r.'); hold on;
polarplot(deg2rad(90-WD),z0med,'k.'); hold off;
ax = gca; ax.RLim = [0 0.3];
title(['Zarnekow Roughness Length, based on data period  ' datestr(ts(1),'yyyy-mm-dd') '  ...  ' datestr(ts(end),'yyyy-mm-dd') ]);
set(gcf,'PaperPositionMode','auto'); 
print([datestr(ts(1),'yyyy') '_roughness_length_' procdate ],'-dpng','-r0');

clear z0 halfbinwidth kk WDwrapped idx1 idx2 ax;


%% initialize parameters of footprint model - part 2

% n KM eq. 36 - exponent of eddy diffusivity power law K(z)=kappa*z^n
n = zeros(halfhours,1);
n(L<0) = (1-24.*zm(L<0)./L(L<0))./(1-16.*zm(L<0)./L(L<0));
n(L>=0) = 1./(1+5.*zm(L>=0)./L(L>=0));

% kappa KM eq. 11 & 32 - constant of eddy diffusivity power law K(z)=kappa*z^n
kappa = k*zm.*ustar./(phi_c.*zm.^n);

% m KM eq. 36 - exponent of wind velocity power law u(z)=U*z^m
m = ustar .* phi_m ./ (k*WS);

% U KM eq. 11 & 31 - constant of wind velocity power law u(z)=U*z^m
U = ustar .* (log(zm./z0med)+psi_m) ./ (k*zm.^m) ;

% r KM p.213 - shape factor
r = 2+m-n;

% µ KM p.213 - constant
mu = (1+m)./r;

% Xi KM eq. 19 - flux length scale
Xi = U.*zm.^r./(r.^2.*kappa);

% aggregate variables for use in footprint equation
gmm = gamma(mu);
mr = m./r;
A = U./(gamma(1./r).*sv).*(kappa.*r.^2./U).^mr;
num = (1/sqrt(2*pi))*Xi.^mu;

% delete not needed stuff
clear phi_m phi_c psi_m z0med n kappa m U r;


%% FOOTPRINT CALCULATION - output data is saved as yearly mat-files

% matrix for cumulating footprint outputs
cumff = zeros(NSdim,EWdim);

% half-hourly output data (timestamp, input data, whole image contribution, class contributions)
KMoutput = nan(halfhours,8 + numclass);

for ii = 1 : halfhours
    
    % create a grid of along-wind (x) and cross-wind (y) distances with respect to tower and wind direction
    wdirm = 90-WD(ii); % change from geografic to mathematic rotation
    x =     cosd(wdirm)*hor - sind(wdirm)*ver;
    y = abs(sind(wdirm)*hor + cosd(wdirm)*ver); % cross wind directions are all positive!
    
    ff2 = zeros(NSdim,EWdim);
    ff2(x>0) = img_res^2*(num(ii).*A(ii).*x(x>0).^(mr(ii)-1))./(x(x>0).^(mu(ii)+1).*exp(Xi(ii)./x(x>0)).*exp(0.5.*(gmm(ii).*y(x>0).*A(ii).*x(x>0).^(mr(ii)-1)).^2));
    
    KMoutput(ii,1:7) = [ts(ii) zm(ii) WS(ii) WD(ii) ustar(ii) sv(ii) L(ii)];
    if isreal(ff2) && ~isnan(sum(ff2(:)))
        cumff = cumff + ff2;
        % image contribution
        KMoutput(ii,8) = sum(ff2(:));
        for nn = 1 : numclass
            % class contributions
            KMoutput(ii,8 + nn) = sum(ff2(class_img == nn));
        end
        disp([datestr(ts(ii),31) '  ' num2str(KMoutput(ii,[3 8]))]);
    end
    %pause;
end

% save results
save([datestr(ts(1),'yyyy') '_cum_2D_footprint_' procdate '.mat'],'cumff');
save([datestr(ts(1),'yyyy') '_footprint_results_' procdate '.mat'],'KMoutput');
% save([datestr(ts(1),'yyyy') '_footprint_workspace_' procdate '.mat']); % uncomment if whole workspace should be saved


%%

