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
    st.title("🔍 POWL Miner")

    st.subheader("Process Mining with the Partially Ordered Workflow Language")

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
            type=["csv", "xes", "gz"],
            help="Supported file types: csv, xes, xes.gz",
        )
        threshold = st.number_input(
            label="Noise Filtering Threshold (0.0 = No Filtering)",
            min_value=0.0,
            max_value=0.99,
            value=0.0,
            step=0.01,
            help="Set the threshold for DFG frequency filtering",
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

                process_model = powl.discover(
                    log, dfg_frequency_filtering_threshold=threshold
                )

                st.session_state["model_gen"] = process_model

            except Exception as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                st.error(body=f"Error during discovery: {e}", icon="⚠️")
                return

    if "model_gen" in st.session_state and st.session_state["model_gen"]:

        st.success("Model generated successfully!", icon="🎉")

        st.write("Export Model")
        powl_model = st.session_state["model_gen"]
        layouted_bpmn = powl.convert_to_bpmn(powl_model)
        try:
            pn, im, fm = powl.convert_to_petri_net(powl_model)

            download_1, download_2 = st.columns(2)
            with download_1:
                bpmn_data = _layout_bpmn(layouted_bpmn).encode("utf-8")

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
                visualization = bpmn_visualizer.apply(
                    layouted_bpmn, parameters={"format": image_format}
                )
                vis_str = visualization.pipe(format="svg").decode("utf-8")

            with st.expander("View Image", expanded=True):
                st.image(vis_str)

        except Exception as e:
            st.error(icon="⚠️", body=str(e))


def footer():
    style = """
        <style>
          .footer-container {
              position: fixed;
              left: 0;
              bottom: 0;
              width: 100%;
              text-align: center;
              padding: 15px 0;
              background-color: white;
              border-top: 2px solid lightgrey;
              z-index: 100;
          }

          .footer-text, .header-text {
              margin: 0;
              padding: 0;
          }
          .footer-links {
              margin: 0;
              padding: 0;
          }
          .footer-links a {
              margin: 0 10px;
              text-decoration: none;
              color: blue;
          }
          .footer-links img {
              vertical-align: middle;
          }
        </style>
        """

    foot = f"""
        <div class='footer-container'>
            <div class='footer-text'>
                Developed by
                <a href="https://www.linkedin.com/in/humam-kourani-98b342232/" target="_blank" style="text-decoration:none;">Humam Kourani</a>
                at the
                <a href="https://www.fit.fraunhofer.de/" target="_blank" style="text-decoration:none;">Fraunhofer Institute for Applied Information Technology FIT</a>.
            </div>
            <div class='footer-links'>
                <a href="https://arxiv.org/abs/2505.07052" target="_blank">
                    <img src="https://img.shields.io/badge/Unlocking%20Non--Block--Structured%20Decisions:%20Inductive%20Mining%20with%20Choice%20Graphs-gray?logo=googledocs&logoColor=white&labelColor=red" alt="Paper">
                </a>
                <a href="mailto:humam.kourani@fit.fraunhofer.de;" target="_blank">
                    <img src="https://img.shields.io/badge/Email-gray?logo=minutemailer&logoColor=white&labelColor=green" alt="Email Humam Kourani">
                </a>
            </div>
        </div>
        """

    st.markdown(style, unsafe_allow_html=True)
    st.markdown(foot, unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(page_title="POWL Miner", page_icon="🔍")
    footer()
    run_app()
