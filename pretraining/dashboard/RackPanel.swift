import AppKit
import CoreGraphics
import WebKit

func fail(_ message: String) -> Never {
    fputs("[panel] \(message)\n", stderr)
    exit(1)
}

func displayID(_ screen: NSScreen) -> CGDirectDisplayID {
    (screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value ?? 0
}

func isRackDisplay(_ screen: NSScreen) -> Bool {
    let id = displayID(screen)
    guard CGDisplayIsMain(id) == 0, CGDisplayVendorNumber(id) == 0x5a64,
          CGDisplayModelNumber(id) == 0x10 else { return false }
    let modes = CGDisplayCopyAllDisplayModes(id, nil) as? [CGDisplayMode] ?? []
    return modes.contains { $0.pixelWidth == 1424 && ($0.pixelHeight == 280 || $0.pixelHeight == 600) }
}

var port = 8766
var requestedID: UInt32?
var checkOnly = false
var listOnly = false
var index = 1
let args = CommandLine.arguments
while index < args.count {
    switch args[index] {
    case "--port":
        index += 1
        guard index < args.count, let value = Int(args[index]), (1024...65535).contains(value) else {
            fail("Invalid port")
        }
        port = value
    case "--display-id":
        index += 1
        guard index < args.count, let value = UInt32(args[index]) else { fail("Invalid display ID") }
        requestedID = value
    case "--check-display": checkOnly = true
    case "--list-displays": listOnly = true
    default: fail("Unknown option: \(args[index])")
    }
    index += 1
}

let screens = NSScreen.screens
if listOnly {
    for screen in screens {
        let id = displayID(screen)
        print("id=\(id) name=\(screen.localizedName) main=\(CGDisplayIsMain(id) != 0) rack=\(isRackDisplay(screen)) desktop=\(Int(screen.frame.width))x\(Int(screen.frame.height))")
    }
    exit(0)
}
let candidates = screens.filter { screen in
    isRackDisplay(screen) && (requestedID == nil || displayID(screen) == requestedID)
}
guard candidates.count == 1, let target = candidates.first else {
    fail("Cannot identify exactly one rack HDMI display. Run with --list-displays; use --display-id if needed. Main display is never used.")
}
if checkOnly {
    print("[panel] Verified rack display id=\(displayID(target)), desktop=\(Int(target.frame.width))x\(Int(target.frame.height)); physical aspect=1424x280. No display settings changed.")
    exit(0)
}

final class PanelWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

final class PanelDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    let targetID: CGDirectDisplayID
    let url: URL
    var window: PanelWindow?
    var keyMonitor: Any?
    var screenObserver: NSObjectProtocol?
    var exitCode: Int32 = 0

    init(screen: NSScreen, port: Int) {
        targetID = displayID(screen)
        url = URL(string: "http://127.0.0.1:\(port)/rack?panel=1424x280")!
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let screen = NSScreen.screens.first(where: { displayID($0) == targetID }),
              isRackDisplay(screen) else { stop("Rack display disconnected"); return }
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
        let view = WKWebView(frame: NSRect(origin: .zero, size: screen.frame.size), configuration: configuration)
        view.navigationDelegate = self
        view.autoresizingMask = [.width, .height]
        view.underPageBackgroundColor = .black
        let panel = PanelWindow(contentRect: screen.frame, styleMask: .borderless,
                                backing: .buffered, defer: false, screen: screen)
        panel.backgroundColor = .black
        panel.isOpaque = true
        panel.hasShadow = false
        panel.isMovable = false
        panel.hidesOnDeactivate = false
        panel.isReleasedWhenClosed = false
        // Cover chrome only on the identified rack screen, not the main desktop.
        panel.level = NSWindow.Level(rawValue: NSWindow.Level.mainMenu.rawValue + 1)
        panel.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary, .ignoresCycle]
        panel.contentView = view
        panel.setFrame(screen.frame, display: true)
        window = panel
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            if event.keyCode == 53 || (event.modifierFlags.contains(.command) && event.charactersIgnoringModifiers == "q") {
                NSApp.terminate(nil)
                return nil
            }
            return event
        }
        screenObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didChangeScreenParametersNotification, object: nil, queue: .main
        ) { [weak self] _ in self?.updateScreen() }
        panel.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        view.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 15))
        print("[panel] Rack-only borderless window. Click it and press Esc or Command-Q to close; Terminal Ctrl-C also closes it.")
        fflush(stdout)
    }

    func updateScreen() {
        // Never migrate an always-on-top panel onto the user's main monitor.
        guard let screen = NSScreen.screens.first(where: { displayID($0) == targetID }),
              isRackDisplay(screen) else { stop("Rack display removed or became main; closing panel"); return }
        window?.setFrame(screen.frame, display: true)
    }

    func stop(_ message: String) {
        fputs("[panel] \(message)\n", stderr)
        exitCode = 1
        NSApp.terminate(nil)
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        let next = navigationAction.request.url
        let allowed = next?.scheme == url.scheme && next?.host == url.host && next?.port == url.port && next?.path == "/rack"
        decisionHandler(allowed ? .allow : .cancel)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        stop("Cannot load the local dashboard: \(error.localizedDescription)")
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        stop("Display renderer exited; launch the panel again when ready")
    }

    func applicationWillTerminate(_ notification: Notification) {
        if let keyMonitor = keyMonitor { NSEvent.removeMonitor(keyMonitor) }
        if let screenObserver = screenObserver { NotificationCenter.default.removeObserver(screenObserver) }
        window?.orderOut(nil)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = PanelDelegate(screen: target, port: port)
app.delegate = delegate
app.run()
exit(delegate.exitCode)
