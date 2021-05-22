#!/usr/bin/env python3

import os
import gi
import sys
import ipdb
import time
import platform
import numpy as np
from rich.console import Console
from datetime import datetime, timezone


gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer


def cb_buffer_probe(pad, info, cb_args):

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer", error=True)
        return
    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    print("features=", features)
    if gstname.find("video") != -1:
        # Get the source bin ghost pad
        bin_ghost_pad = source_bin.get_static_pad("src")
        if not bin_ghost_pad.set_target(decoder_src_pad):
            print("Failed to link decoder src pad to source bin ghost pad", error=True)


def decodebin_child_added(child_proxy, Object, name, user_data):
    print(f"Decodebin child added: {name}")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        print("Unable to create source bin", error=True)

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        print("Unable to create uri decode bin", error=True)
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        print("Failed to add ghost pad in source bin", error=True)
        return None
    return nbin


def make_elm_or_print_err(factoryname, name, printedname):
    """Creates an element with Gst Element Factory make.
    Return the element  if successfully created, otherwise print
    to stderr and return None.
    """
    print("Creating", printedname)
    elm = Gst.ElementFactory.make(factoryname, name)
    if not elm:
        print("Unable to create ", printedname, error=True)
    return elm


def main(
    input_filename: str,
    output_filename: str = None,
):
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        print("Unable to create Pipeline", error=True)

    source_bin = create_source_bin(0, input_filename)

    # Finally encode and save the osd output
    queue = make_elm_or_print_err("queue", "queue", "Queue")

    # Video capabilities: check format and GPU/CPU location
    capsfilter = make_elm_or_print_err("capsfilter", "capsfilter", "capsfilter")
    caps = Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)

    print("Creating MPEG-4 stream")
    encoder = make_elm_or_print_err("avenc_mpeg4", "encoder", "Encoder")
    codeparser = make_elm_or_print_err("mpeg4videoparse", "mpeg4-parser", "Code Parser")

    encoder.set_property("insert-sps-pps", 1)
    encoder.set_property("bitrate", 4e6)

    queue_file = make_elm_or_print_err("queue", "queue_file", "File save queue")
    # codeparser already created above depending on codec
    container = make_elm_or_print_err("qtmux", "qtmux", "Container")
    filesink = make_elm_or_print_err("filesink", "filesink", "File Sink")
    filesink.set_property("location", output_filename)

    pipeline.add(source_bin)
    pipeline.add(streammux)

    pipeline.add(queue)
    pipeline.add(capsfilter)
    pipeline.add(encoder)

    pipeline.add(queue_file)
    pipeline.add(codeparser)
    pipeline.add(container)
    pipeline.add(filesink)

    print("Linking elements in the Pipeline \n")

    # Pipeline Links
    srcpad = source_bin.get_static_pad("src")
    demux_sink = streammux.get_request_pad("sink_0")
    demux_sink.add_probe(Gst.PadProbeType.BUFFER, cb_buffer_probe, None)

    if not srcpad or not demux_sink:
        print("Unable to get file source or mux sink pads", error=True)
    srcpad.link(demux_sink)
    streammux.link(queue)
    queue.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(queue_file.get_static_pad("sink"))

    # Output to File or fake sinks
    queue_file.link(codeparser)
    codeparser.link(container)
    container.link(filesink)


    # GLib loop required for RTSP server
    g_loop = GLib.MainLoop()
    g_context = g_loop.get_context()

    # GStreamer message bus
    bus = pipeline.get_bus()

    # start play back and listen to events
    pipeline.set_state(Gst.State.PLAYING)

    # After setting pipeline to PLAYING, stop it even on exceptions
    try:
        # Custom event loop
        running = True
        while running:
            g_context.iteration(may_block=True)

            message = bus.pop()
            if message is not None:
                t = message.type

                if t == Gst.MessageType.EOS:
                    print("End-of-stream\n")
                    running = False
                elif t == Gst.MessageType.WARNING:
                    err, debug = message.parse_warning()
                    print(f"{err}: {debug}", warning=True)
                elif t == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    print(f"{err}: {debug}", error=True)
                    running = False

        print("Inference main loop ending.")
        pipeline.set_state(Gst.State.NULL)

        print(f"Output file saved: [green bold]{output_filename}[/green bold]")
    except:
        console.print_exception()
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    # Check input arguments
    output_filename = None
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
        print(f"Provided input source: {input_filename}")
        if len(sys.argv) > 2:
            output_filename = sys.argv[2]
            print(f"Save output file: [green]{output_filename}[/green]")
        else:
            output_filename = f"out_{input_filename}"
    else:
        print("Please provide an input file")
        sys.exit(1)

    sys.exit(
        main(
            input_filename=input_filename,
            output_filename=output_filename,
        )
    )
