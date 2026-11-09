import os
import shutil
import tempfile
from enum import Enum

import powl
from powl.visualization.bpmn.layout import layout_bpmn as _layout_bpmn

import streamlit as st

from pm4py.objects.petri_net.exporter.variants.pnml import export_petri_as_string
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer


class ViewType(Enum):
    BPMN = "BPMN"
    POWL = "POWL"
    PETRI = "Petri Net"


def run_app():
    st.title("🔍 PO-Aware POWL Miner")

    st.subheader("A Partial Order Approach to Process Discovery")

    temp_dir = "temp"

    system_dot = shutil.which("dot")
    if system_dot:
        print(f"Found system-wide 'dot' at: {system_dot}")
    else:
        base_path = "/home/adminuser/.conda"
        possible_subpaths = ["bin"]

        for sub in possible_subpaths:
            candidate = os.path.join(base_path, sub, "dot")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                print(f"Found 'dot' at: {candidate}")
                os.environ["PATH"] += os.pathsep + os.path.dirname(candidate)
                break
        else:
            st.error(body="Couldn't find 'dot' — is Graphviz installed?", icon="⚠️")

    with st.form(key="model_gen_form"):

        uploaded_log = st.file_uploader(
            "For **process model discovery**, upload an event log:",
            type=["xes", "gz"],
            help="Supported file types: xes, xes.gz",
        )
        submit_button = st.form_submit_button(label="Run")
        if submit_button:
            if uploaded_log is None:
                st.error(body="No file is selected!", icon="⚠️")
                return
            try:
                contents = uploaded_log.read()
                os.makedirs(temp_dir, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="wb", delete=False, dir=temp_dir, suffix=uploaded_log.name
                ) as temp_file:
                    temp_file.write(contents)
                log = powl.import_event_log(temp_file.name)
                shutil.rmtree(temp_dir, ignore_errors=True)

                process_model = powl.discover_from_partially_ordered_log(log)

                st.session_state["model_gen"] = process_model
            except Exception as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                st.error(body=f"Error during discovery: {e}", icon="⚠️")
                return

    if "model_gen" in st.session_state and st.session_state["model_gen"]:

        st.success("Model generated successfully!", icon="🎉")

        try:
            st.write("Export Model")
            powl_model = st.session_state["model_gen"]
            pn, im, fm = powl.convert_to_petri_net(powl_model)
            bpmn = powl.convert_to_bpmn(powl_model)

            download_1, download_2 = st.columns(2)
            with download_1:
                bpmn_data = _layout_bpmn(bpmn).encode("utf-8")
                st.download_button(
                    label="Download BPMN",
                    data=bpmn_data,
                    file_name="process_model.bpmn",
                    mime="application/xml",
                )

            with download_2:
                pn_data = export_petri_as_string(pn, im, fm)
                st.download_button(
                    label="Download PNML",
                    data=pn_data,
                    file_name="process_model.pnml",
                    mime="application/xml",
                )

            view_option = st.selectbox(
                "Select a view:", [v_type.value for v_type in ViewType]
            )

            image_format = str("svg").lower()
            if view_option == ViewType.POWL.value:
                from powl.visualization.powl import visualizer

                vis_str = visualizer.apply(powl_model)

            elif view_option == ViewType.PETRI.value:
                visualization = pn_visualizer.apply(
                    pn, im, fm, parameters={"format": image_format}
                )
                vis_str = visualization.pipe(format="svg").decode("utf-8")
            else:  # BPMN
                from pm4py.objects.bpmn.layout import layouter

                layouted_bpmn = layouter.apply(bpmn)
                visualization = bpmn_visualizer.apply(
                    layouted_bpmn, parameters={"format": image_format}
                )
                vis_str = visualization.pipe(format="svg").decode("utf-8")

            with st.expander("View Image", expanded=True):
                st.image(vis_str)

        except Exception as e:
            st.error(icon="⚠️", body=str(e))


if __name__ == "__main__":
    st.set_page_config(page_title="PO-Aware POWL Miner", page_icon="🔍")
    # footer()
    run_app()
