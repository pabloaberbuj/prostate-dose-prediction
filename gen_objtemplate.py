"""
Genera un ObjectiveTemplate v1.7 (Eclipse/Helios) a partir de un DVH predicho.
Cada OAR -> una linea (objetivos Type=1 agrupados por Group). PTV -> puntos fijos.
Valida re-parseando y contando objetivos.
"""
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime

def dvh_to_line_points(dose_gy, vol_pct, vol_grid):
    """Muestrea la curva DVH (dosis creciente, vol decreciente) en una grilla de volumen.
    Devuelve lista de (dose_gy, vol_pct) estilo RapidPlan (vol de ~99 -> 0)."""
    # invertir: dosis a la que se alcanza cada nivel de volumen
    # dvh: vol_pct(dose) monotona decreciente -> interpolar dose(vol)
    order = np.argsort(vol_pct)          # vol creciente para interp
    d_at_v = np.interp(vol_grid, vol_pct[order], dose_gy[order])
    return list(zip(d_at_v, vol_grid))

def _obj(parent, typ, op, dose, vol, prio, group, param_a_nil=True):
    o = ET.SubElement(parent, "Objective")
    ET.SubElement(o, "Type").text = str(typ)
    ET.SubElement(o, "Operator").text = str(op)
    ET.SubElement(o, "Dose").text = f"{dose:g}"
    v = ET.SubElement(o, "Volume")
    if vol is None: v.set("{http://www.w3.org/2001/XMLSchema-instance}nil","true")
    else: v.text = f"{vol:g}"
    ET.SubElement(o, "Priority").text = f"{prio:g}"
    pa = ET.SubElement(o, "ParameterA")
    if param_a_nil: pa.set("{http://www.w3.org/2001/XMLSchema-instance}nil","true")
    ET.SubElement(o, "Group").text = str(group)

def _struct(parent, sid, vtype, color=65535):
    s = ET.SubElement(parent, "ObjectivesOneStructure", ID=sid, NAME="", SurfaceOnly="false")
    st = ET.SubElement(s, "StructureTarget")
    for tag in ("VolumeID","VolumeCode"): ET.SubElement(st, tag).text=""
    ET.SubElement(st,"VolumeType").text=vtype
    ET.SubElement(st,"VolumeCodeTable").text=""
    ET.SubElement(s,"Distance").set("{http://www.w3.org/2001/XMLSchema-instance}nil","true")
    ET.SubElement(s,"SamplePoints").set("{http://www.w3.org/2001/XMLSchema-instance}nil","true")
    ET.SubElement(s,"Color").text=str(color)
    return ET.SubElement(s,"StructureObjectives")

def build_template(oar_dvhs, rx_gy, ptv_id="PTV_High",
                   ptv_lower_pct=101.0, ptv_upper_pct=105.0,
                   line_priority=50.0, vis_points=None, template_id="UNet_pred"):
    ET.register_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
    root = ET.Element("ObjectiveTemplate", Version="1.7")
    now = datetime.now().strftime(" %B %d %Y %H:%M:%S:000")
    ET.SubElement(root,"Preview", ID=template_id, Type="Objective",
                  ApprovalStatus="Unapproved", Diagnosis="", TreatmentSite="",
                  AssignedUsers="", Description="", LastModified=now,
                  ApprovalHistory=f"unet Created [{now}]")
    ET.SubElement(root,"Type").text="Helios"
    h=ET.SubElement(root,"Helios",DefaultFixedJaws="false",Interpolate="false",UseColors="false")
    for t,v in [("DefaultSmoothingX","40"),("DefaultSmoothingY","30"),
                ("DefaultMinimizeDose","0"),("DefaultOptimizationType","Beamlet"),
                ("MaxIterations","1000"),("MaxTime","12")]:
        ET.SubElement(h,t).text=v
    nto=ET.SubElement(h,"NormalTissueObjective")
    for t,v in [("Use","true"),("Priority","100"),("DistanceFromTargetBorder","1"),
                ("StartDose","105"),("EndDose","60"),("FallOff","0.05"),("Auto","true")]:
        ET.SubElement(nto,t).text=v
    geos=ET.SubElement(h,"Geos")
    for t,v in [("InitialFieldDistribution","Coplanar"),("MinimumNumberOfFields","5"),
                ("MaximumNumberOfFields","9"),("MaximumElevationAngleForNonCoplanarFields","10"),
                ("MaximumCollimatorVariation","10"),("LocalGeometricOptimizationMode","None")]:
        ET.SubElement(geos,t).text=v
    imat=ET.SubElement(h,"Imat",UseMU="false",JawTracking="false")
    for t,v in [("MUWeight","50"),("MinMU","0"),("MaxMU","2000")]: ET.SubElement(imat,t).text=v

    alls=ET.SubElement(root,"ObjectivesAllStructures")

    # --- OARs: linea Type=1 agrupada ---
    # line_priority: escalar (misma prioridad para todas las lineas OAR, uso
    # historico) o dict {oar_id: prioridad} para prioridad por-estructura (ej.
    # overlap Rectum!PTV con prioridad mas baja que el resto -- ver piloto
    # substructuras, HANDOFF_substructuras_dosis.md).
    grp=10
    for oar_id,(dose_gy,vol_pct) in oar_dvhs.items():
        prio = line_priority[oar_id] if isinstance(line_priority, dict) else line_priority
        so=_struct(alls, oar_id, "Organ")
        vol_grid = np.concatenate([np.arange(99.0, 0.0, -1.0), [0.0]])
        pts = dvh_to_line_points(np.asarray(dose_gy), np.asarray(vol_pct), vol_grid)
        for d,v in pts:
            _obj(so, 1, 0, float(d), float(v), prio, grp)
        if vis_points and oar_id in vis_points:
            for d,v in vis_points[oar_id]:
                _obj(so, 0, 0, float(d), float(v), 0, 0)
        grp+=1

    # --- PTV: puntos fijos ---
    so=_struct(alls, ptv_id, "PTV", color=16776960)
    _obj(so, 0, 1, rx_gy*ptv_lower_pct/100, 100.0, 100, 0)   # lower
    _obj(so, 0, 0, rx_gy*ptv_upper_pct/100, 0.0,   100, 0)   # upper (max)

    return ET.ElementTree(root)

# ---- demo con DVH placeholder (recto/vejiga sinteticos) ----
if __name__=="__main__":
    rx=78.0
    # curva DVH descendente de ejemplo (dosis Gy, vol %)
    d = np.linspace(2, 81, 60)
    rectum_v = 100/(1+np.exp((d-45)/6))      # sigmoide
    bladder_v = 100/(1+np.exp((d-50)/7))
    oar={"Rectum":(d,rectum_v),"Bladder":(d,bladder_v)}
    vis={"Rectum":[(50,50),(60,35),(65,25),(70,20),(75,15)],
         "Bladder":[(65,50),(70,35),(75,25),(80,15)]}
    tree=build_template(oar, rx, vis_points=vis)
    ET.indent(tree, space="")
    tree.write("/home/claude/ObjectiveTemplate_UNet_test.xml", encoding="UTF-8", xml_declaration=True)

    # validar re-parseando
    r=ET.parse("/home/claude/ObjectiveTemplate_UNet_test.xml").getroot()
    ns={"xsi":"http://www.w3.org/2001/XMLSchema-instance"}
    for s in r.iter("ObjectivesOneStructure"):
        objs=s.findall("StructureObjectives/Objective")
        lines=[o for o in objs if o.find("Type").text=="1"]
        pts=[o for o in objs if o.find("Type").text=="0"]
        if objs:
            print(f"{s.get('ID'):10s}  linea(Type1)={len(lines):3d}  puntos(Type0)={len(pts)}")
    print("OK: XML bien formado")
