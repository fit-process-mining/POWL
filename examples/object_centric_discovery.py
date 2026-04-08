import powl
from powl.discovery.object_centric.algorithm import Variants as OC_Variants


def execute_script():

    # Read object centric event log (ocel 1.0 or ocel 2.0)
    object_centric_log = powl.import_ocel(
        r"C:\Users\kourani\OneDrive - Fraunhofer\FIT\oc_powl\oc_logs\recruiting.jsonocel"
    )

    object_centric_pn = powl.discover_petri_net_from_ocel(object_centric_log, variant=OC_Variants.OC_POWL)
    powl.view_ocpn(object_centric_pn)


if __name__ == "__main__":
    execute_script()
