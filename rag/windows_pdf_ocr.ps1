param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [string]$PageNumbers = "",
    [string]$Language = "ru",
    [string]$PreviewDirectory = ""
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime

$storageFileType = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$pdfType = [Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType=WindowsRuntime]
$renderOptionsType = [Windows.Data.Pdf.PdfPageRenderOptions, Windows.Data.Pdf, ContentType=WindowsRuntime]
$streamType = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime]
$decoderType = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$bitmapType = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$pixelType = [Windows.Graphics.Imaging.BitmapPixelFormat, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$alphaType = [Windows.Graphics.Imaging.BitmapAlphaMode, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$ocrType = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$ocrResultType = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType=WindowsRuntime]
$languageType = [Windows.Globalization.Language, Windows.Globalization, ContentType=WindowsRuntime]

$script:operationAsTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq "AsTask" -and $_.IsGenericMethodDefinition -and
    $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1
$script:actionAsTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq "AsTask" -and -not $_.IsGenericMethod -and
    $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.FullName -eq "Windows.Foundation.IAsyncAction"
} | Select-Object -First 1

function Wait-Operation($Operation, [Type]$ResultType) {
    $task = $script:operationAsTask.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    return $task.GetAwaiter().GetResult()
}

function Wait-Action($Action) {
    $task = $script:actionAsTask.Invoke($null, @($Action))
    $null = $task.GetAwaiter().GetResult()
}

$engine = $ocrType::TryCreateFromLanguage($languageType::new($Language))
if ($null -eq $engine) {
    throw "Windows OCR language '$Language' is unavailable. Install its OCR language pack in Windows Settings."
}
$inputFile = Wait-Operation ($storageFileType::GetFileFromPathAsync([IO.Path]::GetFullPath($InputPath))) $storageFileType
$pdf = Wait-Operation ($pdfType::LoadFromFileAsync($inputFile)) $pdfType
if ($PageNumbers -eq "") {
    $selectedPages = @(0..($pdf.PageCount - 1))
} else {
    $selectedPages = @($PageNumbers.Split(',') | ForEach-Object { [int]$_ })
}
if ($PreviewDirectory) {
    [IO.Directory]::CreateDirectory($PreviewDirectory) | Out-Null
}
$results = [System.Collections.Generic.List[object]]::new()
foreach ($number in $selectedPages) {
    if ($number -lt 0 -or $number -ge $pdf.PageCount) { throw "Invalid PDF page index: $number" }
    $page = $pdf.GetPage([uint32]$number)
    $stream = $streamType::new()
    $bitmap = $null
    try {
        $options = $renderOptionsType::new()
        $scale = [Math]::Min(3.0, 3200.0 / [Math]::Max($page.Size.Width, $page.Size.Height))
        $options.DestinationWidth = [uint32][Math]::Max(1, [Math]::Round($page.Size.Width * $scale))
        $options.DestinationHeight = [uint32][Math]::Max(1, [Math]::Round($page.Size.Height * $scale))
        Wait-Action ($page.RenderToStreamAsync($stream, $options))
        $stream.Seek(0)
        $decoder = Wait-Operation ($decoderType::CreateAsync($stream)) $decoderType
        $bitmap = Wait-Operation ($decoder.GetSoftwareBitmapAsync($pixelType::Bgra8, $alphaType::Premultiplied)) $bitmapType
        $result = Wait-Operation ($engine.RecognizeAsync($bitmap)) $ocrResultType
        $text = ($result.Lines | ForEach-Object { $_.Text }) -join "`n"
        $results.Add(@{page=$number; text=$text; language=$engine.RecognizerLanguage.LanguageTag})
        $progressMessage = "OCR page {0}/{1}: {2} characters" -f ($number + 1), $pdf.PageCount, $text.Length
        [Console]::Error.WriteLine($progressMessage)
        if ($PreviewDirectory) {
            $stream.Seek(0)
            $readStream = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($stream)
            $previewPath = Join-Path $PreviewDirectory ("page_{0:D3}.png" -f ($number + 1))
            $outputStream = [IO.File]::Create($previewPath)
            try { $readStream.CopyTo($outputStream) } finally { $outputStream.Dispose() }
        }
    } finally {
        if ($null -ne $bitmap) { $bitmap.Dispose() }
        $stream.Dispose()
        $page.Dispose()
    }
}
$json = ConvertTo-Json -InputObject $results.ToArray() -Depth 4
[IO.File]::WriteAllText($OutputPath, $json, [System.Text.UTF8Encoding]::new($false))
# WinRT keeps COM resources alive beyond PowerShell's pipeline lifetime.
# The script runs in a dedicated child process; exit after flushing the JSON.
[Environment]::Exit(0)
